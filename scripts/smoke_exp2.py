"""Offline end-to-end check: twelve encodings, neural training, decoding and resume."""

import json
import subprocess
import sys
import tempfile
from pathlib import Path


def main():
    repo = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix='exp2-smoke-') as temporary:
        folder = Path(temporary)
        metadata = [{
            'parent_asin': f'item{i:02d}', 'title': f'brand{i % 3} guitar strings material{i % 4}',
            'description': [f'music accessory size{i % 2} steel strings'],
            'price': 5 + i, 'categories': ['Musical Instruments', f'category{i % 4}'],
            'details': {'Brand': f'brand{i % 3}', 'String Gauge': f'gauge{i % 2}'},
        } for i in range(18)]
        reviews = []
        for user in range(24):
            for position in range(8):
                reviews.append({'user_id': f'user{user:02d}',
                                'parent_asin': f'item{(user + position) % 18:02d}',
                                'timestamp': f'2020-01-{position + 1:02d}', 'rating': 3 + position % 3})
            for year, offset in ((2021, 8), (2022, 9)):
                reviews.append({'user_id': f'user{user:02d}',
                                'parent_asin': f'item{(user + offset) % 18:02d}',
                                'timestamp': f'{year}-06-01', 'rating': 5})
        for name, values in (('meta.jsonl', metadata), ('reviews.jsonl', reviews)):
            (folder / name).write_text(''.join(json.dumps(row) + '\n' for row in values), encoding='utf-8')
        source, target = folder / 'exp1', folder / 'exp2'
        config = json.loads((repo / 'configs/experiment1_musical.json').read_text(encoding='utf-8'))
        config.update(metadata=str(folder / 'meta.jsonl'), reviews=str(folder / 'reviews.jsonl'),
                      max_items=18, min_item_interactions=1, min_history=3, max_users=2,
                      max_requests=32, encoding_seeds=[7], scorer_seeds=[7], scorers=['most_popular'],
                      top_k=3, bootstrap_samples=5, budgets=[{'beam': 100}],
                      modes=['preserve'], embeddings={'backend': 'tfidf_svd', 'dim': 6},
                      collaborative_dim=6,
                      tokenizer={'device': 'cpu', 'epochs': 1, 'batch_size': 24, 'branching': 3,
                                 'depth': 2, 'codebook_size': 4, 'latent_dim': 8})
        first_config = folder / 'first.json'
        first_config.write_text(json.dumps(config), encoding='utf-8')
        subprocess.run([sys.executable, '-B', '-m', 'exp1', '--config', str(first_config),
                        '--output', str(source)], cwd=repo, check=True)
        second = {
            'source_run': str(source), 'encodings': config['encodings'], 'encoding_seeds': [7],
            'model_seeds': [7], 'generator': {'device': 'cpu', 'd_model': 8, 'heads': 2,
                'encoder_layers': 1, 'decoder_layers': 1, 'epochs': 1, 'batch_size': 32,
                'max_train_samples': 100, 'dropout': 0., 'max_history': 10},
            'diagnostic_requests': 12, 'diagnostic_seed': 2026, 'top_k': 3,
            'modes': ['preserve', 'local'], 'bootstrap_samples': 5,
            'budgets': [{'beam': 2}, {'beam': 100}, {'beam': 100, 'nodes': 0}],
        }
        second_config = folder / 'second.json'
        second_config.write_text(json.dumps(second), encoding='utf-8')
        command = [sys.executable, '-B', '-m', 'exp2', '--config', str(second_config), '--output', str(target)]
        subprocess.run(command, cwd=repo, check=True)
        rows = [row for path in (target / 'rows').glob('*.json')
                for row in json.loads(path.read_text(encoding='utf-8'))]
        assert len({row['encoding'] for row in rows}) == 12
        assert any(not row['diagnostic'] for row in rows)
        for row in rows:
            assert row['compliance'] in (1., None)
            if row['node_budget'] == 0:
                assert row['scored_nodes'] == 0 and not row['actual']
            elif row['beam'] == 100 and row['diagnostic']:
                assert row['own_oracle_recall'] in (None, 1.), row
                if row['own_search_score_loss'] is not None:
                    assert abs(row['own_search_score_loss']) < 2e-5, row
            if not row['diagnostic']:
                assert row['reference'] is None and row['exact_hr'] is None
        before = {path.name: path.stat().st_mtime_ns for path in (target / 'models').glob('*/generator_best.pt')}
        row_times = {path.name: path.stat().st_mtime_ns for path in (target / 'rows').glob('*.json')}
        # A suite-wide resume first refreshes experiment one's timing/status.
        # Those volatile fields must not invalidate the frozen source identity.
        subprocess.run([sys.executable, '-B', '-m', 'exp1', '--config', str(first_config),
                        '--output', str(source), '--resume'], cwd=repo, check=True)
        subprocess.run(command + ['--resume'], cwd=repo, check=True)
        assert before == {path.name: path.stat().st_mtime_ns for path in (target / 'models').glob('*/generator_best.pt')}
        assert row_times == {path.name: path.stat().st_mtime_ns for path in (target / 'rows').glob('*.json')}
        assert json.loads((target / 'manifest.json').read_text(encoding='utf-8'))['status'] == 'complete'
        print(f'Offline experiment-two integration passed: {len(rows)} rows, 12 encodings, resume verified.')


if __name__ == '__main__':
    main()
