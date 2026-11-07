"""Smoke-прогон: признаки, обе стратегии, метрики. Стадии 3–4, владелец B.

Синтетические данные, никакой сети. Задача теста — поймать сломанный пайплайн
за секунды, а не измерить качество.
"""

import unittest

import numpy as np

from src import evaluate, splits
from src.features import fingerprints

SMILES = [
    "CCO", "CCCC", "c1ccccc1", "C1CCCCC1", "c1ccncc1", "O=C(O)CCl",
    "O=C(Nc1ccccc1)c1ccccc1", "Cc1ccc2ncccc2c1", "CC(=O)Oc1ccccc1C(=O)O",
    "CN1CCC[C@H]1c1cccnc1", "c1ccc2[nH]ccc2c1", "OCC1OC(O)C(O)C(O)C1O",
]


class Features(unittest.TestCase):
    def test_shape_is_stable(self):
        matrix = fingerprints.featurize(SMILES)
        expected = fingerprints.MORGAN_BITS + len(fingerprints.DESCRIPTOR_NAMES)
        self.assertEqual(matrix.shape, (len(SMILES), expected))

    def test_invalid_smiles_does_not_crash_featurization(self):
        matrix = fingerprints.featurize(["CCO", "not a molecule"])
        self.assertEqual(matrix.shape[0], 2)
        self.assertFalse(np.isnan(matrix).any())

    def test_identical_molecules_give_identical_rows(self):
        matrix = fingerprints.featurize(["c1ccccc1", "c1ccccc1"])
        np.testing.assert_array_equal(matrix[0], matrix[1])


class Metrics(unittest.TestCase):
    def test_perfect_prediction_scores_zero_error(self):
        true = np.array([0.0, 1.0, 2.0, 3.0])
        result = evaluate.metrics(true, true)
        self.assertAlmostEqual(result["rmse_log10"], 0.0)
        self.assertAlmostEqual(result["r2_log10"], 1.0)

    def test_shape_mismatch_raises(self):
        with self.assertRaises(ValueError):
            evaluate.metrics(np.array([1.0, 2.0]), np.array([1.0]))

    def test_bootstrap_interval_brackets_the_estimate(self):
        rng = np.random.default_rng(1)
        true = rng.normal(size=120)
        pred = true + rng.normal(scale=0.3, size=120)
        point = evaluate.metrics(true, pred)["rmse_log10"]
        low, high = evaluate.bootstrap_ci(true, pred, n_resamples=200)
        self.assertLess(low, point)
        self.assertGreater(high, point)

    def test_applicability_similarity_is_one_for_a_known_molecule(self):
        train = fingerprints.morgan_matrix(["c1ccccc1", "CCO"])
        query = fingerprints.morgan_matrix(["CCO"])
        similarity = evaluate.applicability_warning(train, query)
        self.assertAlmostEqual(similarity[0], 1.0, places=6)


class SplitsWiring(unittest.TestCase):
    def test_manifest_roundtrip(self):
        import tempfile
        from pathlib import Path

        split = splits.scaffold_split(SMILES)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "splits.json"
            splits.save_manifest(path, split, splits.DEFAULT_SEED, len(SMILES))
            self.assertEqual(splits.load_manifest(path), split)


if __name__ == "__main__":
    unittest.main()
