import tempfile
import unittest
from pathlib import Path

from exp1.__main__ import run


class StageTests(unittest.TestCase):
    def test_invalid_or_conflicting_stage_fails_before_data_reads(self):
        with tempfile.TemporaryDirectory() as temporary:
            for options in ({'stage':'bad'}, {'stage':'encode','stop_after_prepare':True}):
                with self.subTest(options=options), self.assertRaisesRegex(ValueError,'stage'):
                    run({}, Path(temporary), **options)

    def test_staged_search_requires_prior_artifacts_without_training(self):
        config = dict(encodings=['random_balanced'],encoding_seeds=[11],
                      scorers=['most_popular'],scorer_seeds=[11])
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(FileNotFoundError,'search stage.*previous stages'):
                run(config,Path(temporary),resume=True,stage='search')
            self.assertFalse((Path(temporary)/'manifest.json').exists())

    def test_score_requires_complete_encoding_artifacts(self):
        config = dict(encodings=['random_balanced'],encoding_seeds=[11],
                      scorers=['most_popular'],scorer_seeds=[11])
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(FileNotFoundError,'score stage.*previous stages'):
                run(config,Path(temporary),resume=True,stage='score')
