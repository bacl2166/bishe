"""Train generative recommenders and diagnose constrained decoding: python -m exp2."""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
import time
from pathlib import Path

from exp1.__main__ import write_csv, write_json
from exp1.metrics import paired_encoding_comparisons, summarize
from .evaluation import evaluate, select_diagnostics


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _source_files(config):
    source = Path(config['source_run'])
    files = [source / name for name in ('manifest.json', 'prepared.json', 'catalog.json', 'requests.json')]
    for encoding in config['encodings']:
        for seed in config['encoding_seeds']:
            folder = source / 'encodings' / f'{encoding}-{seed}'
            files.extend([folder / 'mapping.json', folder / 'tokenizer_manifest.json'])
    return files


def load_source(config):
    """Reuse precisely the frozen experiment-one catalog, histories and requests."""
    missing = [str(path) for path in _source_files(config) if not path.is_file()]
    if missing:
        raise FileNotFoundError('Run experiment one with the requested encodings first; missing: ' + ', '.join(missing))
    source = Path(config['source_run'])
    data, requests = _read(source / 'prepared.json'), _read(source / 'requests.json')
    ids = [item['id'] for item in data['items']]
    catalog = _read(source / 'catalog.json')
    if not ids or len(set(ids)) != len(ids) or catalog != data['items']:
        raise ValueError('Source catalog and prepared items differ or are invalid.')
    known = set(ids)
    if not requests or len({r['id'] for r in requests}) != len(requests):
        raise ValueError('Source requests must be nonempty with unique IDs.')
    for request in requests:
        if not set(request['feasible']) <= known:
            raise ValueError('Request feasibility is outside the source catalog.')
        if request['history'] != data.get('histories', {}).get(request['user']):
            raise ValueError('Request history differs from the frozen training history.')
    return data, requests


def source_digest(config):
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode())
    digest.update(json.dumps(runtime_versions(), sort_keys=True).encode())
    for package in ('exp1', 'exp2'):
        for path in sorted((Path(__file__).parent.parent / package).glob('*.py')):
            digest.update(f'{package}/{path.name}'.encode())
            digest.update(path.read_bytes())
    for path in _source_files(config):
        digest.update(str(path).encode())
        if path == Path(config['source_run']) / 'manifest.json':
            # Experiment-one resume refreshes status/timing, while its frozen
            # inputs and codes remain identical. Hash identity, not run progress.
            manifest = _read(path)
            stable = {key: manifest.get(key) for key in ('fingerprint', 'config')}
            digest.update(json.dumps(stable, sort_keys=True).encode())
            continue
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
                digest.update(block)
    return digest.hexdigest()


def runtime_versions():
    import torch
    return {'python': platform.python_version(), 'cuda_build': torch.version.cuda,
            'packages': {name: importlib.metadata.version(name) for name in ('numpy', 'torch')}}


def _synchronize(generator):
    if str(generator.device).startswith('cuda'):
        import torch
        torch.cuda.synchronize(generator.device)


def _validate_config(config):
    for field in ('encodings', 'encoding_seeds', 'model_seeds', 'modes', 'budgets'):
        values = config.get(field)
        if not values:
            raise ValueError(f'{field} must not be empty.')
        if field != 'budgets' and len(set(values)) != len(values):
            raise ValueError(f'{field} contains duplicates.')
    if not set(config['modes']) <= {'preserve', 'local'}:
        raise ValueError('Experiment two supports preserve and local probability modes.')
    if int(config.get('top_k', 10)) < 1 or int(config.get('score_batch_size', 256)) < 1:
        raise ValueError('top_k and score_batch_size must be positive.')
    for budget in config['budgets']:
        if budget['beam'] < 1 or (budget.get('nodes') is not None and budget['nodes'] < 0):
            raise ValueError('Beam must be positive and node budgets nonnegative.')


def run(config, output, resume=False):
    from .decoding import Tree, exact_scores, search
    from .model import fit_generator, load_generator

    _validate_config(config)
    data, requests = load_source(config)
    fingerprint = source_digest(config)
    source, output = Path(config['source_run']), Path(output)
    if source.resolve() == output.resolve():
        raise ValueError('Experiment two needs an output directory separate from its source run.')
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / 'manifest.json'
    existing = _read(manifest_path) if manifest_path.exists() else None
    if existing and existing['fingerprint'] != fingerprint:
        raise ValueError('Output belongs to different code/config/source artifacts; choose a new output directory.')
    if existing and not resume:
        raise ValueError('Output already exists; use --resume or choose a new output directory.')
    diagnostics = select_diagnostics(requests, int(config.get('diagnostic_requests', 80)),
                                     int(config.get('diagnostic_seed', 2026)))
    write_json(output / 'diagnostic_requests.json', sorted(diagnostics))
    manifest = {
        'fingerprint': fingerprint, 'config': config, 'status': 'running',
        'python': sys.version, 'platform': platform.platform(), 'source_run': str(source.resolve()),
        'source_fingerprint': _read(source / 'manifest.json')['fingerprint'],
        'catalog_count': len(data['items']), 'requests': len(requests),
        'diagnostic_requests': len(diagnostics),
        'exact_reference': 'exhaustive_feasible_paths_including_eos_without_length_normalization',
        'constraints_in_training': False,
        **runtime_versions(),
    }
    write_json(manifest_path, manifest)
    started = time.perf_counter()
    summaries, job_groups, row_count = [], {}, 0
    request_lookup = {r['id']: r for r in requests}
    batch_size, k = int(config.get('score_batch_size', 256)), int(config.get('top_k', 10))
    for encoding_seed in config['encoding_seeds']:
        for encoding in config['encodings']:
            codes = {item: tuple(code) for item, code in
                     _read(source / 'encodings' / f'{encoding}-{encoding_seed}' / 'mapping.json').items()}
            for model_seed in config['model_seeds']:
                job = f'{encoding}-{encoding_seed}-model-{model_seed}'
                folder, rows_path = output / 'models' / job, output / 'rows' / f'{job}.json'
                job_groups.setdefault((model_seed, encoding_seed), {})[encoding] = rows_path
                if resume and rows_path.is_file():
                    rows = _read(rows_path)
                    row_count += len(rows)
                    summaries.extend(summarize(rows, config.get('bootstrap_samples', 1000)))
                    continue
                print(f'Generator {job}', flush=True)
                training_begin = time.perf_counter()
                if resume and (folder / 'generator_best.pt').is_file():
                    generator = load_generator(data, codes, config.get('generator', {}), model_seed, folder)
                else:
                    generator = fit_generator(data, codes, config.get('generator', {}), model_seed, folder)
                generator.metadata['load_or_fit_seconds'] = time.perf_counter() - training_begin
                generator.metadata['parameter_count'] = sum(p.numel() for p in generator.model.parameters())
                write_json(folder / 'metadata.json', generator.metadata)
                tree = Tree(generator.token_codes)
                rows = []
                for request_index, request in enumerate(requests):
                    feasible, history = set(request['feasible']), request['history']
                    # Run actual search first. No exhaustive leaf scores, reference
                    # ranking or ground-truth positive are available to the decoder.
                    searches = []
                    for mode in config['modes']:
                        for budget in config['budgets']:
                            generator._memory_cache = None
                            _synchronize(generator)
                            begin = time.perf_counter()
                            found = search(generator, tree, history, feasible, mode, k, budget['beam'],
                                           budget.get('nodes'), batch_size=batch_size)
                            _synchronize(generator)
                            searches.append((mode, budget, found, time.perf_counter() - begin))
                    scores, score_seconds = {}, {}
                    if request['id'] in diagnostics:
                        for mode in dict.fromkeys(['preserve', *config['modes']]):
                            generator._memory_cache = None
                            _synchronize(generator)
                            begin = time.perf_counter()
                            scores[mode] = exact_scores(generator, tree, history, feasible, mode,
                                                        batch_size=batch_size)
                            _synchronize(generator)
                            score_seconds[mode] = time.perf_counter() - begin
                    for mode, budget, found, elapsed in searches:
                        metrics = evaluate(found, request['positive'], feasible, k,
                                           scores.get('preserve'), scores.get(mode))
                        raw_reference = metrics['reference'] or []
                        rows.append({
                            'request_id': request['id'],
                            'user': hashlib.sha256(request['user'].encode()).hexdigest()[:20],
                            'scorer': 'encoder_decoder', 'scorer_seed': model_seed,
                            'encoding': encoding, 'encoding_seed': encoding_seed,
                            'family': request['family'], 'strength': request.get('strength', ''),
                            'variant': request.get('variant'), 'matched_request': request.get('matched_request'),
                            'matched_family': request_lookup.get(request.get('matched_request'), {}).get('family'),
                            'mode': mode, 'beam': budget['beam'], 'node_budget': budget.get('nodes'),
                            'feasible_count': len(feasible), 'catalog_count': len(data['items']),
                            'actual': found.items, 'actual_log_scores': found.log_scores,
                            'reference_scores': ({item: scores['preserve'][item] for item in raw_reference}
                                                 if scores else None),
                            'first_pruned': found.pruned,
                            'reference_prefix_counts': ({str(depth): len({tree.paths[item][:depth]
                                                                         for item in raw_reference})
                                                        for depth in range(1, max((len(tree.paths[item])
                                                                                  for item in raw_reference), default=1))}
                                                       if scores else None),
                            'decode_seconds': elapsed, 'exact_seconds': score_seconds.get(mode),
                            **metrics,
                        })
                    if (request_index + 1) % 20 == 0:
                        print(f'{job}: {request_index + 1}/{len(requests)} requests', flush=True)
                write_json(rows_path, rows)
                row_count += len(rows)
                summaries.extend(summarize(rows, config.get('bootstrap_samples', 1000)))
                print(f'Finished {job}: {len(rows)} rows', flush=True)
                del generator
    paired = []
    for paths in job_groups.values():
        if 'random_balanced' not in paths:
            continue
        baseline = _read(paths['random_balanced'])
        for encoding, path in paths.items():
            if encoding != 'random_balanced':
                paired.extend(paired_encoding_comparisons(baseline + _read(path),
                                                         samples=config.get('bootstrap_samples', 1000)))
    write_json(output / 'summary.json', summaries)
    write_csv(output / 'summary.csv', summaries)
    write_json(output / 'paired_comparisons.json', paired)
    write_csv(output / 'paired_comparisons.csv', paired)
    manifest.update(status='complete', result_rows=row_count, wall_seconds=time.perf_counter() - started)
    write_json(manifest_path, manifest)
    print(f'Complete: {output / "summary.csv"}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    config = _read(args.config)
    run(config, args.output or Path(config.get('output', 'runs/experiment2')), args.resume)


if __name__ == '__main__':
    main()
