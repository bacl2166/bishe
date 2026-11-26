"""Suite orchestration tests avoid training and downloads."""
import contextlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASETS = {'musical': 'Musical_Instruments', 'fashion': 'Amazon_Fashion', 'appliances': 'Appliances'}


def load_script(name):
    path = REPO_ROOT / 'scripts' / f'{name}.py'
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ConfigTests(unittest.TestCase):
    def test_expanded_profiles_share_sampling_and_keep_source_chains_separate(self):
        shared = None
        for domain in DATASETS:
            folder = REPO_ROOT / 'configs' / 'expanded'
            first = json.loads((folder / f'experiment1_{domain}.json').read_text())
            second = json.loads((folder / f'experiment2_{domain}.json').read_text())
            previous = json.loads((REPO_ROOT / 'configs' / f'experiment1_{domain}.json').read_text())
            self.assertNotEqual(first['output'], previous['output'])
            self.assertEqual(second['source_run'], first['output'])
            self.assertNotEqual(second['output'], first['output'])
            self.assertEqual(first['encodings'], second['encodings'])
            self.assertEqual(first['evaluation_split'], 'test')
            self.assertEqual(first['min_history'], previous['min_history'])
            self.assertEqual(first['train_before'], previous['train_before'])
            self.assertEqual(first['test_from'], previous['test_from'])
            policy = {k: v for k, v in first.items() if k not in ('output','metadata','reviews')}
            if shared is None:
                shared = policy
            self.assertEqual(shared, policy)

    def test_same_budgets_seeds_and_full_encodings_across_domains(self):
        reference = json.loads((REPO_ROOT / 'configs' / 'experiment1.json').read_text(encoding='utf-8'))
        shared = None
        for dataset, category in DATASETS.items():
            first = json.loads((REPO_ROOT / 'configs' / f'experiment1_{dataset}.json').read_text(encoding='utf-8'))
            second = json.loads((REPO_ROOT / 'configs' / f'experiment2_{dataset}.json').read_text(encoding='utf-8'))
            self.assertEqual(first['encodings'], reference['encodings'])
            self.assertEqual(len(first['encodings']), 12)
            self.assertEqual(second['encodings'], first['encodings'])
            self.assertEqual((first['max_items'], first['max_users'], first['max_requests']), (3000, 200, 800))
            self.assertEqual(first['evaluation_split'], 'test')
            self.assertEqual(first['output'], f'runs/{dataset}-experiment1-v2')
            self.assertEqual(first['scorer']['max_train_samples'], 50000)
            self.assertEqual(first['train_before'], '2021-01-01')
            self.assertEqual(first['test_from'], '2022-01-01')
            self.assertEqual(first['encoding_seeds'], [11, 22, 33])
            self.assertEqual(first['scorer_seeds'], [11, 22, 33])
            self.assertEqual(second['encoding_seeds'], [11, 22, 33])
            self.assertEqual(second['model_seeds'], [11, 22, 33])
            self.assertEqual(first['metadata'], f'data/meta_{category}.jsonl')
            self.assertEqual(first['reviews'], f'data/{category}.jsonl')
            self.assertNotIn('review_prefix_bytes', first)
            self.assertIsNone(first['constraints'].get('spec_field'))
            self.assertEqual(second['source_run'], first['output'])
            self.assertEqual(second['output'], f'runs/{dataset}-experiment2-v2')
            self.assertEqual(second['modes'], ['preserve', 'local'])
            self.assertEqual(second['budgets'], [{'beam': 10}, {'beam': 30}, {'beam': 100}, {'beam': 100, 'nodes': 2000}])
            self.assertEqual((second['diagnostic_requests'], second['diagnostic_seed'], second['top_k'], second['bootstrap_samples']),
                             (80, 2026, 10, 1000))
            self.assertEqual(second['generator'], {'device': 'cuda', 'd_model': 64, 'heads': 2,
                             'encoder_layers': 1, 'decoder_layers': 2, 'epochs': 20, 'patience': 3,
                             'batch_size': 128, 'max_train_samples': 50000, 'lr': .001, 'max_history': 50})
            comparable = {k: v for k, v in first.items() if k not in ('output', 'metadata', 'reviews')}
            if shared is None:
                shared = comparable
            else:
                self.assertEqual(comparable, shared)


class SuiteTests(unittest.TestCase):
    def setUp(self):
        self.suite = load_script('run_suite')
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / 'configs').mkdir()
        for dataset in DATASETS:
            for experiment in (1, 2):
                config = {'source_run': f'runs/{dataset}-experiment1'} if experiment == 2 else {}
                (self.root / 'configs' / f'experiment{experiment}_{dataset}.json').write_text(json.dumps(config), encoding='utf-8')
        self.root_patch = patch.object(self.suite, 'REPO_ROOT', self.root)
        self.root_patch.start()

    def tearDown(self):
        self.root_patch.stop()
        self.tmp.cleanup()

    def run_main(self, args):
        with contextlib.redirect_stdout(io.StringIO()):
            return self.suite.main(args)

    def test_both_orders_by_dataset_and_reuses_current_python_from_any_cwd(self):
        def complete_first(command, **kwargs):
            self.assertEqual(kwargs, {'cwd': self.root, 'check': True})
            self.assertEqual(command[0], sys.executable)
            if command[2] == 'exp1':
                dataset = Path(command[4]).stem.split('_', 1)[1]
                (self.root / 'runs' / f'{dataset}-experiment1').mkdir(parents=True)
        with patch.object(self.suite.subprocess, 'run', side_effect=complete_first) as run:
            original = Path.cwd()
            try:
                os.chdir(self.root.parent)
                self.assertEqual(self.run_main(['--experiment', 'both', '--datasets', 'fashion', 'musical', '--resume']), 0)
            finally:
                os.chdir(original)
        self.assertEqual([c.args[0][2] for c in run.call_args_list], ['exp1', 'exp2', 'exp1', 'exp2'])
        self.assertTrue(all(c.args[0][-1] == '--resume' for c in run.call_args_list))
        self.assertEqual([Path(c.args[0][4]).name for c in run.call_args_list],
                         ['experiment1_fashion.json', 'experiment2_fashion.json', 'experiment1_musical.json', 'experiment2_musical.json'])
        self.assertTrue(all(Path(c.args[0][4]).is_absolute() for c in run.call_args_list))

    def test_prepare_only_never_invokes_experiment2(self):
        with patch.object(self.suite.subprocess, 'run') as run:
            self.run_main(['--experiment', '1', '--datasets', 'appliances', '--prepare-only'])
            self.assertEqual(run.call_count, 1)
            self.assertIn('--prepare-only', run.call_args.args[0])
            self.assertEqual(run.call_args.args[0][2], 'exp1')
        with patch.object(self.suite.subprocess, 'run') as run, contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                self.run_main(['--experiment', 'both', '--prepare-only'])
            run.assert_not_called()

    def test_missing_source_has_clear_error_and_never_starts_training(self):
        stderr = io.StringIO()
        with patch.object(self.suite.subprocess, 'run') as run, contextlib.redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as exc:
                self.run_main(['--experiment', '2', '--datasets', 'fashion'])
        self.assertNotEqual(exc.exception.code, 0)
        self.assertIn('source_run', stderr.getvalue())
        self.assertIn('fashion-experiment1', stderr.getvalue())
        run.assert_not_called()

    def test_failed_child_stops_before_next_dataset(self):
        with patch.object(self.suite.subprocess, 'run', side_effect=subprocess.CalledProcessError(3, ['python'])) as run:
            with self.assertRaises((subprocess.CalledProcessError, SystemExit)):
                self.run_main(['--experiment', '1', '--datasets', 'musical', 'fashion'])
            self.assertEqual(run.call_count, 1)


class DownloadTests(unittest.TestCase):
    def test_all_domain_urls_and_provenance_are_preserved(self):
        downloader = load_script('download_amazon')
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            with patch.object(downloader, 'download') as download:
                for category in DATASETS.values():
                    downloader.main(['--category', category, '--output', str(output)])
            self.assertEqual(download.call_count, 6)
            for call in download.call_args_list:
                url, destination = call.args
                self.assertTrue(url.startswith('https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw/'))
                self.assertIn(destination.name, url)
                self.assertTrue(url.endswith('?download=true'))
            manifest = json.loads((output / 'download_sources.json').read_text(encoding='utf-8'))
            self.assertEqual(len(manifest), 6)
            self.assertTrue(all(f'{c}.jsonl' in manifest and f'meta_{c}.jsonl' in manifest for c in DATASETS.values()))


if __name__ == '__main__':
    unittest.main()
