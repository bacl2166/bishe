"""Offline integration check using synthetic records in Amazon's input schema."""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def main():
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--staged', action='store_true')
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='exp1-smoke-') as temporary:
        folder = Path(temporary)
        metadata, reviews = [], []
        for i in range(24):
            metadata.append({
                'parent_asin': f'item{i:02d}', 'title': f'brand{i % 3} guitar string material{i % 4}',
                'description': [f'music accessory size{i % 2} steel strings'],
                'price': 5 + i, 'categories': ['Musical Instruments', f'category{i % 4}'],
                'details': {'Brand': f'brand{i % 3}', 'String Gauge': f'gauge{i % 2}'},
            })
        for user in range(36):
            for position in range(8):
                reviews.append({'user_id': f'user{user:02d}',
                                'parent_asin': f'item{(user + position) % 24:02d}',
                                'timestamp': f'2020-01-{position + 1:02d}', 'rating': 3 + position % 3})
            for year, offset in ((2021, 8), (2022, 9)):
                reviews.append({'user_id': f'user{user:02d}',
                                'parent_asin': f'item{(user + offset) % 24:02d}',
                                'timestamp': f'{year}-06-01', 'rating': 5})
        for filename, values in (('meta.jsonl', metadata), ('reviews.jsonl', reviews)):
            (folder / filename).write_text(''.join(json.dumps(value) + '\n' for value in values), encoding='utf-8')
        config = json.loads((args.config or repo / 'configs/experiment1.json').read_text(encoding='utf-8'))
        config.update(metadata=str(folder / 'meta.jsonl'), reviews=str(folder / 'reviews.jsonl'),
                      max_items=24, min_item_interactions=1, min_history=3, max_users=3,
                      max_requests=120, encoding_seeds=[7], scorer_seeds=[7], top_k=3,
                      bootstrap_samples=10, budgets=[{'beam': 2}, {'beam': 100}],
                      embeddings={'backend': 'tfidf_svd', 'dim': 8}, collaborative_dim=8,
                      scorer={'device': 'cpu', 'epochs': 1, 'batch_size': 32, 'd_model': 8,
                              'heads': 2, 'layers': 1, 'torch_threads': 2, 'max_train_samples': 100},
                      tokenizer={'device': 'cpu', 'epochs': 1, 'batch_size': 24, 'branching': 3,
                                 'depth': 2, 'codebook_size': 4, 'latent_dim': 8})
        config_path = folder / 'config.json'
        config_path.write_text(json.dumps(config), encoding='utf-8')
        output = folder / 'output'
        command = [sys.executable, '-B', '-m', 'exp1', '--config', str(config_path), '--output', str(output)]
        if args.staged:
            frozen_times = None
            for stage, expected in (('prepare','prepared'),('encode','encoded'),('score','scored'),('search','complete')):
                subprocess.run(command + ['--stage', stage, '--resume'], cwd=repo, check=True)
                status = json.loads((output/'manifest.json').read_text(encoding='utf-8'))['status']
                assert status == expected, (stage, status)
                if stage != 'search':
                    assert not (output/'rows').exists(), stage
                if stage == 'encode':
                    cached = [output/'semantic.npy', output/'collaborative.npy',
                              *sorted((output/'encodings').glob('*/mapping.json'))]
                    frozen_times = {str(p.relative_to(output)):p.stat().st_mtime_ns for p in cached}
                elif stage in ('score','search'):
                    assert frozen_times == {name:(output/name).stat().st_mtime_ns for name in frozen_times}
        else:
            subprocess.run(command, cwd=repo, check=True)
        manifest = json.loads((output / 'manifest.json').read_text(encoding='utf-8'))
        assert manifest['status'] == 'complete'
        rows = []
        for path in (output / 'rows').glob('*.json'):
            rows.extend(json.loads(path.read_text(encoding='utf-8')))
        assert len({row['encoding'] for row in rows}) == 12
        assert len({row['scorer'] for row in rows}) == 3
        assert {row['family'] for row in rows} <= set(config['constraint_families'])
        checkpoint_times = {str(p.relative_to(output)): p.stat().st_mtime_ns
                            for p in (output/'scorers').glob('*/*') if p.is_file()}
        mapping_times = {str(p.relative_to(output)): p.stat().st_mtime_ns
                         for p in (output/'encodings').glob('*/mapping.json')}
        for row in rows:
            if row['beam'] == 100:
                assert row['own_oracle_recall'] in (None, 1.0), row
        subprocess.run(command + ['--resume'], cwd=repo, check=True)
        assert checkpoint_times == {str(p.relative_to(output)): p.stat().st_mtime_ns
                                    for p in (output/'scorers').glob('*/*') if p.is_file()}
        assert mapping_times == {str(p.relative_to(output)): p.stat().st_mtime_ns
                                for p in (output/'encodings').glob('*/mapping.json')}
        print(f"Offline integration passed: {len(rows)} rows, 12 encoders, 3 scorers, resume verified.")


if __name__ == '__main__':
    main()
