import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from exp1.tokenizers import TOKENIZERS, build_codes, _opq, _residual_kmeans


class TokenizerTests(unittest.TestCase):
    def setUp(self):
        self.items = [dict(id=f"i{i:02}", title=f"Camera lens series {i}",
                           description="digital camera" if i % 2 else "outdoor sports",
                           category=["Electronics", "Camera" if i % 2 else "Accessory"],
                           brand=f"brand{i%3}", price=1000*i, specs={"color": ["red", "blue"][i%2]})
                      for i in range(12)]
        rng = np.random.default_rng(4)
        self.semantic = rng.normal(size=(12, 8)).astype(np.float32)
        self.collaborative = rng.normal(size=(12, 5)).astype(np.float32)
        self.config = {"epochs": 1, "depth": 3, "branching": 3,
                       "codebook_size": 4, "opq_iterations": 2, "batch_size": 7}

    def test_all_methods_deterministic_unique_and_permutation_invariant(self):
        with tempfile.TemporaryDirectory() as folder:
            for name in TOKENIZERS:
                with self.subTest(name=name):
                    root = Path(folder) / name
                    a = build_codes(name, self.items, self.semantic, self.collaborative,
                                    self.config, 9, root / "a")
                    reverse = np.arange(11, -1, -1)
                    b = build_codes(name, self.items[::-1], self.semantic[reverse],
                                    self.collaborative[reverse], self.config, 9, root / "b")
                    self.assertEqual(a, b)
                    self.assertEqual(set(a), {item["id"] for item in self.items})
                    self.assertEqual(len(set(a.values())), len(self.items))
                    self.assertTrue(all(row and all(isinstance(t, int) and t >= 0 for t in row)
                                        for row in a.values()))
                    meta = json.loads((root / "a" / "tokenizer_manifest.json").read_text())
                    self.assertFalse(meta["eos_added"])
                    for epoch in meta["details"].get("training_history", []):
                        self.assertTrue(all(np.isfinite(v) for v in epoch.values()))

    def test_empty_duplicates_and_invalid_shapes(self):
        with tempfile.TemporaryDirectory() as folder:
            for items, s, c in [([], self.semantic, self.collaborative),
                                ([self.items[0]]*12, self.semantic, self.collaborative),
                                (self.items, self.semantic[:2], self.collaborative)]:
                with self.assertRaises(ValueError):
                    build_codes("rqkmeans", items, s, c, {}, 1, Path(folder))

    def test_identical_embeddings_collisions_singleton(self):
        with tempfile.TemporaryDirectory() as folder:
            for name in TOKENIZERS:
                with self.subTest(name=name):
                    mapping = build_codes(name, self.items, np.zeros((12, 2)), np.zeros((12, 2)),
                                          dict(self.config, epochs=0), 5, Path(folder) / name)
                    self.assertEqual(len(set(mapping.values())), 12)
                    single = build_codes(name, self.items[:1], self.semantic[:1], self.collaborative[:1],
                                         self.config, 5, Path(folder) / name / "single")
                    self.assertEqual(len(single), 1)

    def test_native_mapping_and_reserved_suffix(self):
        native = {item["id"]: [0] for item in self.items}
        native[self.items[0]["id"]] = [0, 1]
        with tempfile.TemporaryDirectory() as folder:
            mapping = build_codes("letter", self.items, self.semantic, self.collaborative,
                                  {"mode": "native", "native_mapping": native}, 1, Path(folder))
            self.assertEqual(len(set(mapping.values())), 12)
            with self.assertRaises(ValueError):
                build_codes("letter", self.items, self.semantic, self.collaborative,
                            {"mode": "native"}, 1, Path(folder))

    def test_residual_quantization_reduces_error(self):
        codes, books = _residual_kmeans(self.semantic, 3, 3, 1)
        reconstructed = np.zeros_like(self.semantic)
        errors = []
        for level, book in enumerate(books):
            reconstructed += book[codes[:, level]]
            errors.append(float(np.mean((self.semantic-reconstructed)**2)))
        self.assertGreater(errors[0], errors[-1])
        self.assertTrue(all(a >= b-1e-6 for a, b in zip(errors, errors[1:])))

    def test_attribute_first_and_balanced_tree(self):
        with tempfile.TemporaryDirectory() as folder:
            config = dict(self.config, attributes=["brand", "price", "specs.color"])
            codes = build_codes("attribute_first", self.items, self.semantic,
                                self.collaborative, config, 3, Path(folder) / "attr")
            for i in range(9):
                self.assertEqual(codes[f"i{i:02}"][0], codes[f"i{i+3:02}"][0])
            for name in ["random_balanced", "seater"]:
                codes = build_codes(name, self.items, self.semantic, self.collaborative,
                                    self.config, 3, Path(folder) / name)
                counts = list(__import__("collections").Counter(c[0] for c in codes.values()).values())
                self.assertLessEqual(max(counts)-min(counts), 1)

    def test_opq_rotation_is_orthogonal(self):
        with tempfile.TemporaryDirectory() as folder:
            build_codes("opq", self.items, self.semantic, self.collaborative,
                        self.config, 2, Path(folder))
            with np.load(Path(folder) / "opq_model.npz") as model:
                rotation = model["rotation"]
                np.testing.assert_allclose(rotation.T @ rotation, np.eye(len(rotation)), atol=1e-5)


if __name__ == "__main__":
    unittest.main()
