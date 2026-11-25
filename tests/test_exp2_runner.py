import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from exp2.__main__ import load_source, source_digest


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.config = {'source_run': str(self.root), 'encodings': ['random_balanced'],
                       'encoding_seeds': [7], 'model_seeds': [7]}

    def write(self, relative, value):
        target = self.root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(value), encoding='utf-8')

    def fixture(self):
        self.write('manifest.json', {'status': 'complete', 'fingerprint': 'fixture'})
        self.write('prepared.json', {'items': [{'id': 'a'}, {'id': 'b'}],
                                    'histories': {'u': ['a']}, 'train_sequences': [['a', 'b']]})
        self.write('catalog.json', [{'id': 'a'}, {'id': 'b'}])
        self.write('requests.json', [{'id': 'r', 'user': 'u', 'history': ['a'],
                                     'positive': 'b', 'feasible': ['a', 'b']}])
        self.write('encodings/random_balanced-7/mapping.json', {'a': [0], 'b': [1]})
        self.write('encodings/random_balanced-7/tokenizer_manifest.json', {'mode': 'adapted'})

    def test_missing_artifacts_explain_exp1_prerequisite(self):
        with self.assertRaisesRegex(FileNotFoundError, 'experiment one'):
            load_source(self.config)

    def test_inputs_and_mapping_changes_invalidate_digest(self):
        self.fixture()
        self.assertEqual(len(load_source(self.config)[1]), 1)
        first = source_digest(self.config)
        self.write('encodings/random_balanced-7/mapping.json', {'a': [0], 'b': [2]})
        self.assertNotEqual(first, source_digest(self.config))

    def test_feasible_outside_catalog_rejected(self):
        self.fixture()
        self.write('requests.json', [{'id': 'r', 'user': 'u', 'history': ['a'], 'feasible': ['future']}])
        with self.assertRaisesRegex(ValueError, 'catalog'):
            load_source(self.config)

    def test_environment_versions_invalidate_resume_digest(self):
        self.fixture()
        first = source_digest(self.config)
        with patch('exp2.__main__.runtime_versions', return_value={'python': 'another', 'packages': {}}):
            self.assertNotEqual(first, source_digest(self.config))

    def test_source_progress_changes_do_not_break_resume(self):
        self.fixture()
        first = source_digest(self.config)
        self.write('manifest.json', {'status': 'running', 'wall_seconds': 123., 'fingerprint': 'fixture'})
        self.assertEqual(first, source_digest(self.config))
        self.write('manifest.json', {'status': 'complete', 'wall_seconds': 0., 'fingerprint': 'changed'})
        self.assertNotEqual(first, source_digest(self.config))

    def test_changed_request_history_rejected(self):
        self.fixture()
        self.write('requests.json', [{'id': 'r', 'user': 'u', 'history': ['b'], 'feasible': ['a']}])
        with self.assertRaisesRegex(ValueError, 'history'):
            load_source(self.config)
