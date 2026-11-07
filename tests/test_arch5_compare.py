import unittest

import numpy as np
import pandas as pd

from src.arch5_compare import exact_logs, fit_residuals, public_training_rows, scaffold_folds


class SourceHierarchyTests(unittest.TestCase):
    def test_benchmark_excludes_held_structures_and_scores_original_rows_once(self):
        from unittest.mock import patch
        from src.arch5_compare import benchmark
        smiles = ['c1ccccc1', 'c1ccncc1', 'C1CCCCC1', 'C1CCCC1', 'C1CCC1',
                  'C1CC1', 'c1ccc2ccccc2c1', 'c1ccoc1', 'c1ccsc1']
        frame = pd.DataFrame({'standard_smiles': smiles})
        reference = pd.DataFrame({'standard_smiles': [*smiles, smiles[0]],
            'original_row': np.arange(10), 'IC50': np.ones(10),
            'CC50': np.ones(10)*10, 'SI': np.ones(10)*10})
        matrix = {s: np.array([i, i*i], dtype=float) for i, s in enumerate(smiles)}
        class FakeHead:
            def __init__(self, x):
                self.seen = set(x[:, 0])
            def predict_log_concentrations(self, x):
                if self.seen & set(x[:, 0]):
                    raise AssertionError('Held structure leaked into public fit')
                return {'IC50': np.zeros(len(x)), 'CC50': np.ones(len(x))}
            def predict_si(self, x):
                self.predict_log_concentrations(x)
                return np.ones(len(x))*10
        def fit(rows, x, params):
            return FakeHead(x)
        with patch('src.arch5_compare.fit_public', side_effect=fit), patch('src.arch5_compare.fit_kaggle', side_effect=fit):
            report, oof = benchmark(frame, reference, frame, matrix, folds=3)
        self.assertEqual(report['n_scored_original_rows'], 10)
        self.assertEqual(oof.original_row.nunique(), 10)
        self.assertEqual(len(oof), 10)
        self.assertFalse(report['holdout_used'])

    def test_inner_oof_keeps_unrelated_public_scaffolds(self):
        from unittest.mock import patch
        from src.arch5_compare import public_oof
        reference = pd.DataFrame({'standard_smiles': ['c1ccccc1', 'c1ccncc1', 'C1CCCCC1']})
        public = pd.DataFrame({'standard_smiles': [*reference.standard_smiles, 'C1CCCC1']})
        features = {s: np.zeros(2) for s in public.standard_smiles}
        used = []
        class FakeHead:
            def predict_log_concentrations(self, x):
                return {'IC50': np.zeros(len(x)), 'CC50': np.ones(len(x))}
        def fit(rows, matrix, params):
            used.append(set(rows.standard_smiles))
            return FakeHead()
        with patch('src.arch5_compare.fit_public', side_effect=fit):
            result = public_oof(reference, public, features, None, 3, 41)
        self.assertEqual(len(used), 3)
        self.assertTrue(all('C1CCCC1' in rows for rows in used))
        self.assertTrue(all(len(rows) == 3 for rows in used))
        self.assertTrue(np.isfinite(result['IC50']).all())

    def test_public_training_excludes_every_row_on_held_scaffold(self):
        public = pd.DataFrame({"standard_smiles": ["c1ccccc1", "Cc1ccccc1", "c1ccncc1"]})
        allowed = {"c1ccccc1"}
        selected = public_training_rows(public, allowed)
        self.assertEqual(selected.standard_smiles.tolist(), ["c1ccccc1", "Cc1ccccc1"])

    def test_kaggle_outer_folds_keep_murcko_scaffolds_intact(self):
        smiles = ["c1ccccc1", "Cc1ccccc1", "c1ccncc1", "C1CCCCC1", "CC1CCCCC1", "C1CCCC1"]
        folds = scaffold_folds(smiles, folds=3, seed=41)
        destinations = {}
        for fold, indices in folds.items():
            for index in indices:
                from src.splits import murcko_scaffold
                scaffold = murcko_scaffold(smiles[index])
                self.assertIn(scaffold, ("c1ccccc1", "c1ccncc1", "C1CCCCC1", "C1CCCC1"))
                self.assertEqual(destinations.setdefault(scaffold, fold), fold)
        self.assertGreaterEqual(len(destinations), 3)

    def test_kaggle_endpoint_targets_stay_original_and_mask_censored_values(self):
        rows = pd.DataFrame({"IC50": [10., 100., np.nan], "IC50_qualifier": ["", ">", ""]})
        np.testing.assert_allclose(exact_logs(rows, "IC50")[[0]], [1.])
        self.assertTrue(np.isnan(exact_logs(rows, "IC50")[1:]).all())
        # Original competition rows keep their separate SI targets; no median is made here.
        reference = pd.DataFrame({"IC50": [10., 100.], "CC50": [20., 300.], "SI": [2., 3.]})
        self.assertEqual(reference.SI.tolist(), [2., 3.])

    def test_small_residual_head_learns_log_endpoint_correction(self):
        x = np.arange(12, dtype=float).reshape(6, 2)
        baseline = {"IC50": np.ones(6), "CC50": np.ones(6) * 2}
        reference = pd.DataFrame({"IC50": 10 ** (1 + .1*x[:, 0]),
                                  "CC50": 10 ** (2 - .2*x[:, 1])})
        models = fit_residuals(x, reference, baseline, alpha=1e-8)
        np.testing.assert_allclose(models["IC50"].predict(x), .1*x[:, 0], atol=1e-5)
        np.testing.assert_allclose(models["CC50"].predict(x), -.2*x[:, 1], atol=1e-5)


if __name__ == "__main__":
    unittest.main()
