import json
import tempfile
import unittest
from pathlib import Path

from amazon_experiment import PilotData, build_pilot_encodings, fit_models, prepare_data, read_jsonl, run_pilot


class AmazonExperimentTest(unittest.TestCase):
    def test_review_prefix_drops_only_its_incomplete_final_line(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "review.jsonl"
            first_row = b'{"item": "A"}\n'
            path.write_bytes(first_row + b'{"item": "B"}\n')
            self.assertEqual(list(read_jsonl(path, len(first_row) + 4)), [{"item": "A"}])
            with self.assertRaises(ValueError):
                list(read_jsonl(path, 1000))

    def test_catalog_and_user_histories_are_train_only(self):
        def review(user, item, timestamp):
            return {"user_id": user, "parent_asin": item, "timestamp": timestamp}

        metadata = [
            {"parent_asin": "A", "price": "10.00"},
            {"parent_asin": "B", "price": 20.00},
            {"parent_asin": "NEW", "price": 5},
            {"parent_asin": "MISSING", "price": "None"},
        ]
        reviews = [
            review("u", "B", 12), review("u", "A", 2), review("u", "B", 1),
            review("v", "A", 3), review("v", "NEW", 10), review("v", "B", 11),
            review("u", "MISSING", 4),
        ]
        with tempfile.TemporaryDirectory() as folder:
            meta_path, review_path = Path(folder) / "meta.jsonl", Path(folder) / "review.jsonl"
            for path, rows in ((meta_path, metadata), (review_path, reviews)):
                path.write_text("\n".join(map(json.dumps, rows)), encoding="utf-8")
            data = prepare_data(meta_path, review_path, cutoff=10, max_items=10)

        self.assertEqual(data.prices, {"A": 1000, "B": 2000})
        self.assertEqual(data.train_counts, {"A": 2, "B": 1})
        self.assertEqual(data.histories["u"], ("B", "A"))
        self.assertEqual(data.test_items, {"u": "B", "v": "NEW"})

    def test_item_knn_is_personalized_and_popularity_is_not(self):
        counts = {"A": 2, "B": 4, "C": 5}
        histories = {"u": ("A", "B"), "v": ("C", "B"), "w": ("C",)}
        pop, knn = fit_models(counts, histories)
        self.assertEqual(pop(("A",)), pop(("C",)))
        self.assertGreater(knn(("A",))["B"], knn(("A",))["C"])
        self.assertGreater(knn(("C",))["B"], knn(("A",))["C"])
        self.assertNotEqual(knn(("A",)), knn(("C",)))

    def test_encoding_comparison_matches_group_sizes(self):
        encodings = build_pilot_encodings({str(i): 100 + i for i in range(11)}, groups=4, seed=42)

        def group_sizes(codes):
            return [sum(code[0] == group for code in codes.values()) for group in range(1, 5)]

        self.assertEqual(group_sizes(encodings["price_ordered"]), [3, 3, 3, 2])
        self.assertEqual(group_sizes(encodings["random_balanced"]), [3, 3, 3, 2])
        self.assertEqual(len(set(encodings["price_ordered"].values())), 11)

    def test_pilot_keeps_out_of_catalog_labels_in_coverage_denominator(self):
        data = PilotData(
            prices={"A": 1000, "B": 2000, "C": 3000},
            train_counts={"A": 2, "B": 2, "C": 2},
            histories={"u": ("A", "B", "C"), "v": ("C", "B", "A")},
            test_items={"u": "B", "v": "NEW"},
            priced_items=3,
            review_rows=8,
        )
        result = run_pilot(data, max_users=10, groups=2, seed=42, k=2, widths=[2])
        self.assertEqual(result["price_cap_cents"], 2000)
        self.assertEqual(result["test_requests"], 2)
        self.assertEqual(result["positive_feasible_rate"], 0.5)
        self.assertEqual(len(result["results"]), 6)
        for row in result["results"]:
            self.assertEqual(row["hr"], 1)
            self.assertEqual(row["oracle_recall"], 1)


if __name__ == "__main__":
    unittest.main()
