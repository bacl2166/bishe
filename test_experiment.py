import unittest

from experiment import (
    CATALOG,
    SCORES,
    build_encodings,
    constrained_search,
    evaluate,
    exact_top_k,
    feasible_items,
)


class ExperimentTest(unittest.TestCase):
    def setUp(self):
        self.encodings = build_encodings(CATALOG, seed=42)
        self.feasible = feasible_items(CATALOG, max_price_cents=2000)

    def test_price_cap_is_inclusive(self):
        self.assertEqual(feasible_items(CATALOG, 1900), {"P2", "P3"})
        self.assertEqual(feasible_items(CATALOG, 0), set())

    def test_each_encoding_is_unique_and_reproducible(self):
        self.assertEqual(len(self.encodings), 3)
        self.assertEqual(self.encodings, build_encodings(CATALOG, seed=42))
        for codes in self.encodings.values():
            self.assertEqual(set(codes), set(SCORES))
            self.assertEqual(len(set(codes.values())), len(CATALOG))

    def test_narrow_beam_exposes_a_search_miss(self):
        by_type = constrained_search(
            self.encodings["product_type"], SCORES, self.feasible, beam_width=1, k=2
        )
        by_price = constrained_search(
            self.encodings["price_band"], SCORES, self.feasible, beam_width=1, k=2
        )
        self.assertEqual(by_type.items, ("P2",))
        self.assertEqual(by_price.items, ("P3", "P2"))

    def test_wide_beam_recovers_exact_ranking_for_every_encoding(self):
        reference = exact_top_k(SCORES, self.feasible, k=2)
        self.assertEqual(reference, ("P3", "P2"))
        for name, codes in self.encodings.items():
            with self.subTest(encoding=name):
                result = constrained_search(codes, SCORES, self.feasible, beam_width=5, k=2)
                self.assertEqual(result.items, reference)

    def test_mask_skips_groups_without_feasible_items(self):
        result = constrained_search(
            self.encodings["price_band"], SCORES, self.feasible, beam_width=1, k=2
        )
        self.assertEqual(result.scored_nodes, 3)
        empty = constrained_search(
            self.encodings["price_band"], SCORES, set(), beam_width=1, k=2
        )
        self.assertEqual(empty.items, ())
        self.assertEqual(empty.scored_nodes, 0)

    def test_metrics_distinguish_a_miss_from_an_infeasible_label(self):
        miss = evaluate(("P2",), ("P3", "P2"), self.feasible, positive="P3")
        self.assertEqual((miss.hr, miss.ndcg, miss.oracle_recall), (0, 0.0, 0.5))
        ranked_second = evaluate(("P2", "P3"), ("P3", "P2"), self.feasible, positive="P3")
        self.assertEqual(ranked_second.hr, 1)
        self.assertAlmostEqual(ranked_second.ndcg, 0.6309297536)
        excluded = evaluate(("P2",), ("P2",), {"P2"}, positive="P3")
        self.assertEqual((excluded.hr, excluded.ndcg, excluded.oracle_recall), (None, None, 1.0))
        self.assertIsNone(evaluate((), (), set(), positive="P3").oracle_recall)


if __name__ == "__main__":
    unittest.main()
