import unittest
from types import SimpleNamespace

from exp1.metrics import summarize
from exp2.evaluation import evaluate, select_diagnostics


class EvaluationTests(unittest.TestCase):
    def found(self, items):
        return SimpleNamespace(items=items, scored_nodes=9, expanded_nodes=4, budget_exhausted=False)

    def test_diagnostics_do_not_use_positives_or_input_order(self):
        requests = [{'id': str(i), 'positive': 'p'} for i in range(20)]
        chosen = select_diagnostics(requests, 7, 17)
        self.assertEqual(len(chosen), 7)
        altered = [{**r, 'positive': 'changed'} for r in reversed(requests)]
        self.assertEqual(chosen, select_diagnostics(altered, 7, 17))

    def test_objective_shift_separate_from_search_error(self):
        raw = {'a': -1., 'b': -2., 'c': -3.}
        local = {'a': -3., 'b': -2., 'c': -1.}
        row = evaluate(self.found(['c', 'b']), 'a', set(raw), 2, raw, local)
        self.assertEqual(row['oracle_recall'], .5)
        self.assertEqual(row['own_oracle_recall'], 1.)
        self.assertEqual(row['own_search_score_loss'], 0.)
        self.assertGreater(row['search_score_loss'], 0.)
        self.assertEqual(row['objective_lost_hit'], 1)
        self.assertEqual(row['own_lost_hit'], 0)
        self.assertEqual(row['positive_original_rank'], 1)
        self.assertEqual(row['positive_own_rank'], 3)

    def test_nondiagnostic_metrics_are_null_and_summarize_safely(self):
        row = evaluate(self.found(['a']), 'a', {'a', 'b'}, 1)
        self.assertEqual(row['hr'], 1)
        self.assertIsNone(row['oracle_recall'])
        self.assertIsNone(row['lost_hit'])
        row.update(user='u')
        result = summarize([row], bootstrap_samples=5)[0]
        self.assertEqual(result['hr'], 1)
        self.assertIsNone(result['lost_hit_rate'])
        self.assertIsNone(result['user_mean_ndcg_delta'])

    def test_empty_and_infeasible_labels(self):
        row = evaluate(self.found([]), 'a', set(), 10, {}, {})
        self.assertIsNone(row['hr'])
        self.assertIsNone(row['oracle_recall'])
        self.assertIsNone(row['compliance'])
        self.assertEqual(row['shortfall'], 0)
        self.assertEqual(row['empty_feasible'], 1)
        row = evaluate(self.found(['a']), 'outside', {'a'}, 10, {'a': -1}, {'a': -1})
        self.assertIsNone(row['exact_hr'])
        self.assertEqual(row['oracle_recall'], 1.)

    def test_incomplete_score_maps_fail(self):
        with self.assertRaises(ValueError):
            evaluate(self.found(['a']), 'a', {'a', 'b'}, 2, {'a': -1}, {'a': -1})
