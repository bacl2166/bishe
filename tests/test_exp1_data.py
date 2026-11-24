import copy
import gzip
import json
import tempfile
import unittest
from pathlib import Path
from exp1.data import prepare, read_jsonl
from exp1.constraints import build_requests, matches


class DataTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.meta = self.root / 'meta.jsonl'
        self.reviews = self.root / 'reviews.jsonl'
        self.meta_rows = [
            {'parent_asin': 'a', 'price': '10.25', 'categories': ['Music', 'Strings'], 'average_rating': 5,
             'rating_number': 10000, 'details': {'Brand': 'X', 'String Gauge': 'Light'}},
            {'parent_asin': 'b', 'price': None, 'categories': ['Music', 'Strings'],
             'details': {'Brand': 'Y', 'String Gauge': 'Heavy'}},
            {'parent_asin': 'c', 'price': 20, 'categories': ['Music', 'Guitars'],
             'details': {'Brand': 'X', 'String Gauge': 'Light'}},
            {'parent_asin': 'future', 'price': 1},
        ]
        self.rows = [self.row('u', 'a', '2019-01-01', 1), self.row('u', 'b', '2019-02-01', 2),
                     self.row('u', 'c', '2020-01-01', 3), self.row('train_only', 'a', '2020-02-01', 5),
                     self.row('u', 'future', '2021-02-01', 5), self.row('u', 'b', '2021-01-01', 5),
                     self.row('u', 'future', '2022-01-01', 5), self.row('u', 'a', '2022-02-01', 5)]
        self.write(self.meta, self.meta_rows)
        self.write(self.reviews, self.rows)
        self.config = {'metadata': str(self.meta), 'reviews': str(self.reviews), 'max_items': 3,
                       'min_history': 3, 'request_seed': 31, 'max_requests': 100}

    @staticmethod
    def row(user, item, timestamp, rating):
        return dict(user_id=user, parent_asin=item, timestamp=timestamp, rating=rating)

    @staticmethod
    def write(path, rows):
        path.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')

    def tearDown(self):
        self.tmp.cleanup()

    def load(self, **kwargs):
        return prepare({**self.config, **kwargs}, self.root / 'out')

    def test_chronological_freeze_train_ratings_and_all_training_users(self):
        data = self.load()
        items = {i['id']: i for i in data['items']}
        self.assertEqual(set(items), {'a', 'b', 'c'})
        self.assertEqual(items['a']['price'], 1025)
        self.assertIsNone(items['b']['price'])
        self.assertEqual((items['a']['rating_mean'], items['a']['rating_count']), (3, 2))
        self.assertEqual(items['a']['specs']['string_gauge'], 'light')
        self.assertEqual(data['histories'], {'u': ['a', 'b', 'c']})
        self.assertEqual(data['validation'][0]['positive'], 'b')
        self.assertEqual(data['test'][0]['positive'], 'future')
        self.assertFalse(data['test'][0]['in_catalog'])
        self.assertEqual(data['test'][0]['history'], data['validation'][0]['history'])
        self.assertIn(['a'], data['train_sequences'])
        self.assertTrue((self.root / 'out' / 'prepared.json').exists())

    def test_future_cannot_change_train_and_catalog_ties_are_stable(self):
        before = self.load()
        self.write(self.reviews, self.rows + [self.row('train_only', 'future', '2025-01-01', 5)] * 50)
        after = self.load()
        self.assertEqual(before['items'], after['items'])
        self.assertEqual(before['train_sequences'], after['train_sequences'])
        limited = self.load(max_items=2, min_history=1)
        self.assertEqual([i['id'] for i in limited['items']], ['a', 'b'])

    def test_json_errors_prefix_and_gzip(self):
        raw = self.root / 'sample.raw'
        valid = b'{"a": 1}\n'
        raw.write_bytes(valid + b'{"broken":')
        with self.assertRaisesRegex(ValueError, 'line 2'):
            list(read_jsonl(raw))
        self.assertEqual(list(read_jsonl(raw, raw.stat().st_size)), [{'a': 1}])
        gz = self.root / 'sample.jsonl.gz'
        with gzip.open(gz, 'wb') as stream:
            stream.write(valid)
        self.assertEqual(list(read_jsonl(gz)), [{'a': 1}])

    def test_metadata_duplicates_fail_explicitly(self):
        self.write(self.meta, self.meta_rows + self.meta_rows[:1])
        with self.assertRaisesRegex(ValueError, 'duplicate metadata item a'):
            self.load()

    def test_rule_label_independence_seed_and_exact_random_counts(self):
        data = self.load()
        before = build_requests(data, self.config)
        modified = copy.deepcopy(data)
        modified['test'][0]['positive'] = 'a'
        after = build_requests(modified, {**self.config, 'encoding_seed': 999})
        stable = lambda rows: [(r['id'], r['family'], r['rule'], r['feasible']) for r in rows]
        self.assertEqual(stable(before), stable(after))
        self.assertEqual(stable(before), stable(build_requests(data, self.config)))
        lookup = {r['id']: r for r in before}
        for r in before:
            self.assertEqual(r['feasible'], [i['id'] for i in data['items'] if matches(i, r['rule'])])
            if r['family'] == 'random_matched':
                self.assertEqual(len(r['feasible']), len(lookup[r['matched_request']]['feasible']))
        self.assertIn('price_brand_spec', {r['family'] for r in before})

    def test_predicate_or_and_unknown_and_empty_diagnostics(self):
        item = {'id': 'a', 'price': 100, 'brand': 'X', 'category': ['Music'], 'specs': {}}
        self.assertTrue(matches(item, {'brand': ['Y', 'X'], 'price': {'min': 100, 'max': 200}}))
        self.assertFalse(matches(item, {'brand': ['X'], 'price': {'max': 99}}))
        self.assertFalse(matches({**item, 'price': None}, {'price': {'max': 200}}))
        self.assertFalse(matches(item, {'spec': {'size': ['large']}}))
        data = self.load()
        requests = build_requests(data, {**self.config, 'constraint_families': ['rating', 'unsupported'], 'rating_min_count': 999})
        self.assertEqual(len(requests), 3)
        self.assertTrue(all(not r['feasible'] for r in requests))
        self.assertEqual(data['audit']['constraints']['skipped_families'][0]['family'], 'unsupported')


if __name__ == '__main__':
    unittest.main()
