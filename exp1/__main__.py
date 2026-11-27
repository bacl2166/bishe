"""Run with python -m exp1 --config configs/experiment1.json."""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

from .metrics import measure, paired_encoding_comparisons, summarize
from .search import Tree, prepare_distribution, search


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temporary, path)


def write_csv(path, rows):
    fields = sorted({field for row in rows for field in row})
    with Path(path).open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value) if isinstance(value, (list, dict)) else value
                             for key, value in row.items()})


def semantic_embeddings(items, config, seed):
    texts = [str(item.get('title', '')) + ' ' + str(item.get('description', '')) for item in items]
    if config.get('backend', 'sentence_transformer') == 'sentence_transformer':
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(config.get('model', 'sentence-transformers/all-MiniLM-L6-v2'),
                                    device=config.get('device', 'cpu'))
        return model.encode(texts, batch_size=config.get('batch_size', 64),
                            normalize_embeddings=True, show_progress_bar=True)
    if config['backend'] != 'tfidf_svd':
        raise ValueError('Unknown semantic embedding backend.')
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    matrix = TfidfVectorizer(max_features=config.get('max_features', 20000),
                            token_pattern=r'(?u)\b\w+\b').fit_transform(texts)
    dim = min(config.get('dim', 64), max(1, len(items) - 1), max(1, matrix.shape[1] - 1))
    if matrix.shape[1] < 2:
        embedding = matrix.toarray()
    else:
        embedding = TruncatedSVD(dim, random_state=seed).fit_transform(matrix)
    return embedding / np.maximum(np.linalg.norm(embedding, axis=1, keepdims=True), 1e-12)


def probability(logits, temperature):
    values = np.asarray(logits, dtype=np.float64) / temperature
    if not np.all(np.isfinite(values)):
        raise ValueError('Nonfinite scorer output.')
    values -= values.max()
    weights = np.exp(np.maximum(values, -690))
    return weights / weights.sum()


def input_digest(config):
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode())
    for source in sorted(Path(__file__).parent.glob('*.py')):
        digest.update(source.name.encode())
        digest.update(source.read_bytes())
    for options in [config.get('tokenizer', {}), *config.get('tokenizer_overrides', {}).values()]:
        native = options.get('native_mapping')
        if isinstance(native, str):
            digest.update(Path(native).read_bytes())
    for field in ('metadata', 'reviews'):
        path = Path(config.get('data', {}).get(field, config.get(field, '')))
        if not path.is_file():
            raise FileNotFoundError(f'Missing data input {field}: {path}')
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
                digest.update(block)
    return digest.hexdigest()


def _check_stage_inputs(config, output, stage):
    required = ['manifest.json', 'prepared.json', 'requests.json', 'audit.json']
    if stage in ('score', 'search'):
        required += ['semantic.npy', 'collaborative.npy', 'catalog.json']
        for encoding in config['encodings']:
            for seed in config['encoding_seeds']:
                required += [f'encodings/{encoding}-{seed}/{name}'
                             for name in ('mapping.json', 'tokenizer_manifest.json')]
    if stage == 'search':
        for scorer in config['scorers']:
            for seed in config['scorer_seeds']:
                required += [f'scorers/{scorer}-{seed}/{name}'
                             for name in ('probabilities.npz', 'metadata.json')]
    missing = [name for name in required if not (output / name).is_file()]
    if missing:
        raise FileNotFoundError(f'{stage} stage requires previous stages; missing: ' + ', '.join(missing))


def run(config, output, resume=False, stop_after_prepare=False, stage=None):
    if stage not in (None, 'all', 'prepare', 'encode', 'score', 'search'):
        raise ValueError(f'Unknown stage: {stage}')
    if stop_after_prepare and stage not in (None, 'prepare'):
        raise ValueError('--prepare-only conflicts with the requested stage')
    stage = 'prepare' if stop_after_prepare else stage or 'all'
    if stage in ('encode', 'score', 'search'):
        _check_stage_inputs(config, output, stage)
        if not resume:
            raise ValueError(f'{stage} stage requires --resume to reuse previous stages')
    from .constraints import build_requests
    from .data import prepare
    from .scorers import collaborative_embeddings, fit_scorer
    from .tokenizers import build_codes

    output.mkdir(parents=True, exist_ok=True)
    fingerprint = input_digest(config)
    manifest_path = output / 'manifest.json'
    existing = json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.exists() else None
    if existing and existing['fingerprint'] != fingerprint:
        raise ValueError('Output belongs to a different config/data version. Choose a new output directory.')
    if existing and not resume:
        raise ValueError('Output already exists. Use --resume or choose a new output directory.')
    manifest = {'fingerprint': fingerprint, 'config': config, 'python': sys.version,
                'requested_stage': stage,
                'platform': platform.platform(), 'status': 'running'}
    manifest['packages'] = {
        name: importlib.metadata.version(name)
        for name in ('numpy', 'scipy', 'scikit-learn', 'torch', 'sentence-transformers')
    }
    write_json(manifest_path, manifest)
    started = time.perf_counter()
    prepared_path = output / 'prepared.json'
    if resume and prepared_path.exists():
        data = json.loads(prepared_path.read_text(encoding='utf-8'))
    else:
        print('Preparing train-only catalog and chronological requests', flush=True)
        data = prepare(config, output)
        write_json(prepared_path, data)
    requests = build_requests(data, config)
    if not requests:
        raise ValueError('No requests. Inspect data audit, attributes and min_history.')
    write_json(output / 'requests.json', requests)
    request_lookup = {request['id']: request for request in requests}
    write_json(output / 'audit.json', data['audit'])
    print(f"Catalog {len(data['items'])}; constraint requests {len(requests)}", flush=True)
    if stage == 'prepare':
        manifest.update(status='prepared', wall_seconds=time.perf_counter() - started)
        write_json(manifest_path, manifest)
        return

    embeddings_config = config.get('embeddings', {'backend': 'sentence_transformer'})
    embedding_path = output / 'semantic.npy'
    if resume and embedding_path.exists():
        semantic = np.load(embedding_path)
    else:
        semantic = semantic_embeddings(data['items'], embeddings_config, config.get('request_seed', 2026))
        temporary = embedding_path.with_suffix('.npy.tmp')
        with temporary.open('wb') as stream:
            np.save(stream, semantic)
        os.replace(temporary, embedding_path)
    collaborative_path = output / 'collaborative.npy'
    if resume and collaborative_path.exists():
        collab = np.load(collaborative_path)
    else:
        collab = collaborative_embeddings(data, config.get('collaborative_dim', 64), config.get('request_seed', 2026))
        temporary = collaborative_path.with_suffix('.npy.tmp')
        with temporary.open('wb') as stream:
            np.save(stream, collab)
        os.replace(temporary, collaborative_path)
    write_json(output / 'catalog.json', data['items'])
    ids = [item['id'] for item in data['items']]
    id_index = {item: index for index, item in enumerate(ids)}
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate catalog IDs.')
    trees = {}
    for seed in config['encoding_seeds']:
        for encoding in config['encodings']:
            folder = output / 'encodings' / f'{encoding}-{seed}'
            mapping_path = folder / 'mapping.json'
            if resume and mapping_path.exists() and (folder / 'tokenizer_manifest.json').exists():
                codes = {item: tuple(code) for item, code in json.loads(mapping_path.read_text(encoding='utf-8')).items()}
            else:
                print(f'Encoding {encoding}, seed {seed}', flush=True)
                tokenizer_config = {**config.get('tokenizer', {}), **config.get('tokenizer_overrides', {}).get(encoding, {})}
                codes = build_codes(encoding, data['items'], semantic, collab, tokenizer_config, seed, folder)
                write_json(mapping_path, codes)
            if set(codes) != set(ids):
                raise ValueError(f'{encoding} does not cover exactly the shared catalog.')
            tree = Tree({item: codes[item] for item in ids})
            trees[(encoding, seed)] = tree
            write_json(folder / 'structure.json', {
                'items': len(ids), 'nodes': len(tree.descendants),
                'max_code_length': max(len(codes[item]) for item in ids),
                'mean_code_length': float(np.mean([len(codes[item]) for item in ids])),
                'max_branching': max(map(len, tree.children.values())),
            })

    if stage == 'encode':
        manifest.update(status='encoded', wall_seconds=time.perf_counter() - started)
        write_json(manifest_path, manifest)
        print(f'Encoding stage complete: {len(trees)} mappings', flush=True)
        return

    summary_rows = []
    row_count = 0
    job_groups = {}
    for scorer_seed in config['scorer_seeds']:
        for scorer_name in config['scorers']:
            scorer_folder = output / 'scorers' / f'{scorer_name}-{scorer_seed}'
            scorer_folder.mkdir(parents=True, exist_ok=True)
            probability_path = scorer_folder / 'probabilities.npz'
            probabilities = {}
            if resume and probability_path.exists():
                with np.load(probability_path, allow_pickle=False) as archive:
                    probabilities = {tuple(json.loads(key)): values
                                     for key, values in zip(archive['history_keys'], archive['probabilities'])}
            else:
                print(f'Fitting {scorer_name}, seed {scorer_seed}', flush=True)
                scorer = fit_scorer(scorer_name, data, config.get('scorer', {}), scorer_seed, scorer_folder)
                write_json(scorer_folder / 'metadata.json', scorer.metadata)
                for request in requests:
                    key = tuple(request['history'])
                    if key not in probabilities:
                        probabilities[key] = probability(scorer.scores(request['history']), config.get('temperature', 1.0))
                temporary = probability_path.with_suffix('.npz.tmp')
                with temporary.open('wb') as stream:
                    np.savez(stream, history_keys=np.array([json.dumps(key) for key in probabilities]),
                             probabilities=np.stack(list(probabilities.values())))
                os.replace(temporary, probability_path)
            if stage == 'score':
                continue
            for (encoding, encoding_seed), tree in trees.items():
                job_path = output / 'rows' / f'{scorer_name}-{scorer_seed}-{encoding}-{encoding_seed}.json'
                job_groups.setdefault((scorer_name, scorer_seed, encoding_seed), {})[encoding] = job_path
                if resume and job_path.exists():
                    rows = json.loads(job_path.read_text(encoding='utf-8'))
                    row_count += len(rows)
                    summary_rows.extend(summarize(rows, config.get('bootstrap_samples', 1000)))
                    continue
                rows = []
                for request in requests:
                    history_key = tuple(request['history'])
                    q = probabilities[history_key]
                    feasible = set(request['feasible'])
                    if not feasible <= set(ids):
                        raise ValueError('Request feasibility is outside the fixed catalog.')
                    k = config.get('top_k', 10)
                    original_ranking = sorted(feasible, key=lambda item: (-q[id_index[item]], item))
                    original = original_ranking[:k]
                    tracked = set(original) | ({request['positive']} & feasible)
                    for mode in config['modes']:
                        begin = time.perf_counter()
                        dist = prepare_distribution(tree, q, feasible, mode)
                        preparation_seconds = time.perf_counter() - begin
                        for budget in config['budgets']:
                            begin = time.perf_counter()
                            found = search(dist, k, budget['beam'], budget.get('nodes'), tracked)
                            decoding_seconds = time.perf_counter() - begin
                            metrics = measure(dist, original, found, request['positive'], k)
                            user = hashlib.sha256(request['user'].encode()).hexdigest()[:20]
                            rows.append({
                                'request_id': request['id'], 'user': user,
                                'scorer': scorer_name, 'scorer_seed': scorer_seed,
                                'encoding': encoding, 'encoding_seed': encoding_seed,
                                'family': request['family'], 'strength': request.get('strength', ''),
                                'variant': request.get('variant'),
                                'matched_request': request.get('matched_request'),
                                'matched_family': request_lookup.get(request.get('matched_request'), {}).get('family'),
                                'mode': mode, 'beam': budget['beam'], 'node_budget': budget.get('nodes'),
                                'feasible_count': len(feasible), 'catalog_count': len(ids),
                                'actual': found.items, 'reference': original, 'own_reference': dist.top(k),
                                'reference_prefix_counts': {
                                    str(depth): len({tree.paths[item][:depth] for item in original})
                                    for depth in range(1, max((len(tree.paths[item]) for item in original), default=1))
                                },
                                'positive_original_rank': (original_ranking.index(request['positive']) + 1
                                                           if request['positive'] in feasible else None),
                                'first_pruned': found.pruned,
                                'preparation_seconds': preparation_seconds,
                                'decode_seconds': decoding_seconds,
                                **metrics,
                            })
                write_json(job_path, rows)
                row_count += len(rows)
                summary_rows.extend(summarize(rows, config.get('bootstrap_samples', 1000)))
                print(f'Finished {job_path.name}: {len(rows)} rows', flush=True)
    if stage == 'score':
        manifest.update(status='scored', wall_seconds=time.perf_counter() - started)
        write_json(manifest_path, manifest)
        print('Scoring stage complete: fixed catalog probabilities cached', flush=True)
        return
    paired = []
    for paths in job_groups.values():
        if 'random_balanced' not in paths:
            continue
        baseline = json.loads(paths['random_balanced'].read_text(encoding='utf-8'))
        for encoding, path in paths.items():
            if encoding != 'random_balanced':
                arm = json.loads(path.read_text(encoding='utf-8'))
                paired.extend(paired_encoding_comparisons(
                    baseline + arm, samples=config.get('bootstrap_samples', 1000)))
    write_json(output / 'summary.json', summary_rows)
    write_json(output / 'paired_comparisons.json', paired)
    write_csv(output / 'summary.csv', summary_rows)
    write_csv(output / 'paired_comparisons.csv', paired)
    manifest.update(status='complete', result_rows=row_count, wall_seconds=time.perf_counter() - started)
    write_json(manifest_path, manifest)
    print(f'Complete: {output / "summary.json"}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--stage', choices=('prepare', 'encode', 'score', 'search', 'all'))
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8'))
    if config.get('temperature', 1.0) <= 0:
        parser.error('temperature must be positive')
    output = args.output or Path(config.get('output', 'runs/experiment1'))
    run(config, output, args.resume, args.prepare_only, args.stage)


if __name__ == '__main__':
    main()
