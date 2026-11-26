import copy
import json
import tempfile
import unittest
from pathlib import Path

from exp1.audit import run, summarize_requests


def request(key, user, family='brand', positive=True, feasible=('a',), **extra):
    return dict(id=key, user=user, family=family, strength='tight', variant='one',
                positive_feasible=positive, positive_in_catalog=True,
                feasible=list(feasible), **extra)


class AuditTests(unittest.TestCase):
    def test_repeated_requests_do_not_inflate_independent_users(self):
        rows = [request(str(i), 'same') for i in range(40)]
        group = summarize_requests(rows, min_users=10, min_positive_users=30)[0]
        self.assertEqual(group['positive_feasible_requests'], 40)
        self.assertEqual(group['positive_feasible_users'], 1)
        self.assertEqual(group['recommendation_support'], 'limited')

    def test_random_controls_stratified_by_source_and_empty_sets_kept(self):
        rows = [request('b', 'u', positive=False, feasible=()),
                request('p', 'u', family='price'),
                request('rb', 'u', family='random_matched', positive=False, feasible=(), matched_request='b'),
                request('rp', 'u', family='random_matched', matched_request='p')]
        groups = summarize_requests(rows, min_users=1, min_positive_users=1)
        controls = [g for g in groups if g['family'] == 'random_matched']
        self.assertEqual({g['matched_family'] for g in controls}, {'brand', 'price'})
        brand = next(g for g in groups if g['family'] == 'brand')
        self.assertEqual(brand['empty_feasible_rate'], 1.)
        self.assertEqual(brand['recommendation_support'], 'no_feasible_positive')

    def test_validation_audit_ignores_test_label_changes_and_preserves_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            data = dict(items=[{'id':'a', 'train_count':5}], histories={'u':['a']},
                        validation=[{'user':'u','positive':'a'}],
                        test=[{'user':'u','positive':'outside'}], audit={})
            config = dict(request_seed=7,max_requests=10,constraint_families=['no_filter'])
            manifest = {'config':config,'fingerprint':'frozen','status':'prepared'}
            (folder/'manifest.json').write_text(json.dumps(manifest),encoding='utf-8')
            (folder/'audit.json').write_text('original audit',encoding='utf-8')
            path = folder/'prepared.json'
            path.write_text(json.dumps(data),encoding='utf-8')
            first = run(folder, split='validation', min_users=1, min_positive_users=1)
            changed = copy.deepcopy(data)
            changed['test'][0]['positive'] = 'a'
            path.write_text(json.dumps(changed),encoding='utf-8')
            second = run(folder, split='validation', min_users=1, min_positive_users=1)
            self.assertEqual(first['groups'],second['groups'])
            self.assertEqual((folder/'audit.json').read_text(),'original audit')
            self.assertTrue((folder/'constraint-support-validation.csv').exists())
            self.assertEqual(second['groups'][0]['recommendation_support'],'meets_planning_count')

    def test_invalid_thresholds_and_missing_random_source_fail(self):
        with self.assertRaises(ValueError):
            summarize_requests([], min_positive_users=0)
        with self.assertRaisesRegex(ValueError,'matched'):
            summarize_requests([request('r','u',family='random_matched',matched_request='missing')])
