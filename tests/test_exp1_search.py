import unittest

import numpy as np

from exp1.search import Tree, prepare_distribution, search


class SearchTest(unittest.TestCase):
    def setUp(self):
        self.tree = Tree({'a': (1, 1), 'b': (1, 2), 'c': (2, 1), 'd': (2, 2)})
        self.q = np.array([.5, .1, .2, .2])
        self.feasible = {'b', 'c', 'd'}

    def test_mask_normalization_changes_complete_item_scores(self):
        raw = prepare_distribution(self.tree, self.q, self.feasible, 'preserve')
        local = prepare_distribution(self.tree, self.q, self.feasible, 'local')
        ideal = prepare_distribution(self.tree, self.q, self.feasible, 'feasible_mass')
        self.assertEqual(raw.top(2), ['c', 'd'])
        self.assertEqual(local.top(2), ['b', 'c'])
        self.assertEqual(ideal.top(2), ['c', 'd'])
        np.testing.assert_allclose(np.exp(local.log_scores[[1, 2, 3]]), [.6, .2, .2])

    def test_full_search_and_variable_terminal_paths_match_enumeration(self):
        for mode in ('preserve', 'local', 'feasible_mass'):
            dist = prepare_distribution(self.tree, self.q, self.feasible, mode)
            self.assertEqual(search(dist, 2, 10).items, dist.top(2))
        tree = Tree({'a': (1,), 'b': (1, 2), 'c': (2, 1)})
        dist = prepare_distribution(tree, np.array([.1, .7, .2]), {'a', 'b', 'c'}, 'local')
        self.assertEqual(search(dist, 3, 10).items, dist.top(3))

    def test_small_beam_and_empty_set(self):
        dist = prepare_distribution(self.tree, self.q, self.feasible, 'preserve')
        result = search(dist, 2, 1)
        self.assertEqual(result.items, ['b'])
        self.assertEqual(result.pruned['c'], 1)
        empty = prepare_distribution(self.tree, self.q, set(), 'local')
        self.assertEqual(search(empty, 10, 2).items, [])

    def test_global_targets_preserve_equal_score_ties_across_unequal_paths(self):
        codes = {'a': (1, 9, 5), 'b': (1, 3), 'c': (2,), 'd': (1, 9, 7, 5), 'e': (4, 5)}
        tree = Tree(codes)
        q = np.array([.1, .2, .2, .2, .3])
        for mode in ('preserve', 'feasible_mass'):
            dist = prepare_distribution(tree, q, {'a', 'b', 'c', 'd'}, mode)
            self.assertEqual(dist.top(3), ['b', 'c', 'd'])
            self.assertEqual(search(dist, 3, 50).items, ['b', 'c', 'd'])

    def test_local_without_filtering_preserves_uniform_ties(self):
        rng = np.random.default_rng(0)
        codes = {f'i{i:03}': tuple(rng.integers(0, 5, size=int(rng.integers(1, 8)))) + (i,)
                 for i in range(100)}
        dist = prepare_distribution(Tree(codes), np.ones(100), set(codes), 'local')
        self.assertEqual(dist.top(10), [f'i{i:03}' for i in range(10)])


if __name__ == '__main__':
    unittest.main()
