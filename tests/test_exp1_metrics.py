import unittest

import numpy as np

from exp1.metrics import measure, summarize
from exp1.search import Tree, prepare_distribution, search


class MetricsTest(unittest.TestCase):
    def test_log_score_loss_separates_objective_shift_and_search_and_handles_empty_output(self):
        tree = Tree({'a': (1,1), 'b': (1,2), 'c': (2,1), 'd': (2,2)})
        dist = prepare_distribution(tree, [.5,.1,.2,.2], {'b','c','d'}, 'local')
        row = measure(dist, ['c','d'], search(dist,2,1), 'c', 2)
        self.assertAlmostEqual(row['search_score_loss'], np.log(2.))
        self.assertAlmostEqual(row['own_search_score_loss'], 0.)
        empty = measure(dist, ['c','d'], search(dist,2,1,node_budget=1), 'c', 2)
        self.assertIsNone(empty['search_score_loss'])
        self.assertIsNone(empty['own_search_score_loss'])
        self.assertEqual(empty['shortfall'], 1)

    def test_target_shift_and_search_loss_have_different_references(self):
        tree = Tree({'a': (1, 1), 'b': (1, 2), 'c': (2, 1), 'd': (2, 2)})
        dist = prepare_distribution(tree, [.5, .1, .2, .2], {'b', 'c', 'd'}, 'local')
        found = search(dist, 2, 1)
        row = measure(dist, ['c', 'd'], found, 'c', 2)
        self.assertEqual(row['oracle_recall'], 0)
        self.assertEqual(row['own_oracle_recall'], .5)
        self.assertEqual(row['objective_topk_overlap'], .5)
        self.assertGreater(row['distribution_kl'], 0)
        self.assertEqual(row['lost_hit'], 1)
        self.assertEqual(row['own_lost_hit'], 1)

    def test_zero_exact_hits_has_undefined_loss_rate(self):
        tree = Tree({'a': (1,), 'b': (2,)})
        dist = prepare_distribution(tree, [.9, .1], {'a', 'b'}, 'preserve')
        row = measure(dist, ['a'], search(dist, 1, 3), 'b', 1)
        row.update(user='u', scorer='test', encoding='random_balanced')
        aggregate = summarize([row], bootstrap_samples=10)[0]
        self.assertIsNone(aggregate['lost_hit_rate'])
        self.assertEqual(aggregate['evaluated_positives'], 1)
        self.assertEqual(aggregate['hr'], 0)

    def test_node_budget_is_a_hard_ceiling(self):
        tree = Tree({'a': (1, 1), 'b': (2, 1)})
        dist = prepare_distribution(tree, np.array([.9, .1]), {'a', 'b'}, 'preserve')
        found = search(dist, 2, 10, node_budget=1)
        self.assertTrue(found.budget_exhausted)
        self.assertLessEqual(found.scored_nodes, 1)


if __name__ == '__main__':
    unittest.main()
