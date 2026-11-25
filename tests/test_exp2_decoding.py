"""Neural token decoding uses conditional vocabulary probabilities, including EOS."""
import unittest
from math import log

import numpy as np

from exp2.decoding import Tree, exact_scores, search


class ConditionalGenerator:
    def __init__(self):
        self.token_codes = {'a': (3, 2), 'b': (4, 5, 2), 'c': (4, 6, 2), 'd': (7, 2)}
        self.probabilities = {
            (): {3: .3, 4: .4, 7: .1},
            (3,): {2: .1},
            (4,): {5: .2, 6: .7},
            (4, 5): {2: .9},
            (4, 6): {2: .1},
            (7,): {2: .8},
        }
        self.calls = []

    def log_probs(self, history, prefixes):
        self.calls.append((list(history), list(prefixes)))
        output = []
        for prefix in prefixes:
            specified = self.probabilities[prefix]
            remainder = (1 - sum(specified.values())) / (10 - len(specified))
            values = np.full(10, remainder)
            for token, probability in specified.items():
                values[token] = probability
            output.append(np.log(values))
        return np.asarray(output)


class DecodingTests(unittest.TestCase):
    def setUp(self):
        self.generator = ConditionalGenerator()
        self.tree = Tree(self.generator.token_codes)
        self.feasible = set(self.generator.token_codes)

    def test_preserve_keeps_full_vocabulary_mass_and_eos(self):
        scores = exact_scores(self.generator, self.tree, ['h'], self.feasible, 'preserve', batch_size=2)
        for item, expected in {'a': .03, 'b': .072, 'c': .028, 'd': .08}.items():
            self.assertAlmostEqual(scores[item], log(expected))
        called = [prefix for history, prefixes in self.generator.calls for prefix in prefixes]
        self.assertCountEqual(called, [(), (3,), (4,), (7,), (4, 5), (4, 6)])
        self.assertTrue(all(history == ['h'] and len(prefixes) <= 2
                            for history, prefixes in self.generator.calls))

    def test_local_renormalizes_only_children_with_feasible_descendants(self):
        scores = exact_scores(self.generator, self.tree, [], self.feasible, 'local')
        for item, expected in {'a': .375, 'b': 1 / 9, 'c': 7 / 18, 'd': .125}.items():
            self.assertAlmostEqual(scores[item], log(expected))
        subset = exact_scores(self.generator, self.tree, [], {'a', 'b', 'd'}, 'local')
        self.assertAlmostEqual(subset['b'], log(.5))
        self.assertNotIn('c', subset)

    def test_wide_beam_recovers_each_modes_exact_order_and_scores(self):
        for mode, expected in [('preserve', ['d', 'b', 'a', 'c']), ('local', ['c', 'a', 'd', 'b'])]:
            with self.subTest(mode=mode):
                exact = exact_scores(self.generator, self.tree, [], self.feasible, mode)
                found = search(self.generator, self.tree, [], self.feasible, mode, 4, 20, batch_size=2)
                self.assertEqual(found.items, expected)
                for item in expected:
                    self.assertAlmostEqual(found.log_scores[item], exact[item])
                self.assertEqual(found.scored_nodes, 9)
                self.assertEqual(found.expanded_nodes, 6)
                self.assertFalse(found.budget_exhausted)

    def test_beam_prunes_prefixes_using_neural_scores(self):
        found = search(self.generator, self.tree, [], self.feasible, 'preserve', 4, 1)
        self.assertEqual(found.items, ['c'])
        self.assertAlmostEqual(found.log_scores['c'], log(.028))

    def test_budget_is_strict_and_stops_before_an_atomic_layer(self):
        for budget, scored, items in [(0, 0, []), (2, 0, []), (6, 3, []), (7, 7, ['d', 'a'])]:
            with self.subTest(budget=budget):
                self.generator.calls.clear()
                found = search(self.generator, self.tree, [], self.feasible, 'preserve', 4, 20, budget)
                self.assertEqual(found.scored_nodes, scored)
                self.assertEqual(found.items, items)
                self.assertTrue(found.budget_exhausted)
                self.assertLessEqual(found.scored_nodes, budget)
                if not scored:
                    self.assertFalse(self.generator.calls)

    def test_completed_eos_does_not_compete_with_unfinished_beam(self):
        generator = ConditionalGenerator()
        generator.token_codes = {'short': (3, 2), 'long': (3, 4, 2)}
        generator.probabilities = {(): {3: .6}, (3,): {2: .4, 4: .5}, (3, 4): {2: .9}}
        tree = Tree(generator.token_codes)
        found = search(generator, tree, [], set(tree.ids), 'preserve', 2, 1)
        self.assertEqual(found.items, ['long', 'short'])
        self.assertAlmostEqual(found.log_scores['short'], log(.24))
        self.assertAlmostEqual(found.log_scores['long'], log(.27))

    def test_empty_and_singleton_feasible(self):
        self.assertEqual(exact_scores(self.generator, self.tree, [], set()), {})
        empty = search(self.generator, self.tree, [], set(), 'local', 3, 1)
        self.assertEqual(empty.items, [])
        self.assertEqual(empty.scored_nodes, 0)
        self.assertFalse(self.generator.calls)
        singleton = search(self.generator, self.tree, [], {'b'}, 'local', 3, 1)
        self.assertEqual(singleton.items, ['b'])
        self.assertAlmostEqual(singleton.log_scores['b'], 0.)

    def test_ties_sort_by_item_id(self):
        tree = Tree({'z': (3, 2), 'a': (4, 2)})
        self.generator.probabilities = {(): {3: .4, 4: .4}, (3,): {2: .5}, (4,): {2: .5}}
        found = search(self.generator, tree, [], set(tree.ids), 'preserve', 2, 4)
        self.assertEqual(found.items, ['a', 'z'])

    def test_rejects_unknown_modes_ids_and_invalid_counts(self):
        for mode in ['feasible_mass', 'bad']:
            with self.assertRaises(ValueError):
                exact_scores(self.generator, self.tree, [], set(), mode)
        with self.assertRaises(ValueError):
            exact_scores(self.generator, self.tree, [], {'unknown'})
        for options in [{'k': 0, 'beam_width': 1}, {'k': 1, 'beam_width': 0},
                        {'k': 1, 'beam_width': 1, 'node_budget': -1},
                        {'k': 1, 'beam_width': 1, 'batch_size': 0}]:
            with self.assertRaises(ValueError):
                search(self.generator, self.tree, [], self.feasible, 'preserve', **options)

    def test_invalid_paths_and_nonfinite_generator_output_fail(self):
        for codes in [{}, {'a': (3,)}, {'a': (0, 2)}, {'a': (1, 2)},
                      {'a': (2, 3, 2)}, {'a': (3, 2), 'b': (3, 2)}]:
            with self.subTest(codes=codes), self.assertRaises(ValueError):
                Tree(codes)
        self.generator.log_probs = lambda history, prefixes: np.full((len(prefixes), 10), np.nan)
        with self.assertRaises(ValueError):
            exact_scores(self.generator, self.tree, [], self.feasible)


if __name__ == '__main__':
    unittest.main()
