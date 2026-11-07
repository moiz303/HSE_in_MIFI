import unittest
import numpy as np
import pandas as pd
from unittest.mock import patch
from tests.test_xgb_hybrid import HybridWorkflow as _HybridWorkflow
from src.data.kaggle_reference import restore_targets, original_pairs
from src.hybrid_cv import choose_weight, classifier_labels, select_on_train


class OriginalTargets(unittest.TestCase):
    def test_reference_overrides_mixed_targets_without_inventing_ic50(self):
        frame = pd.DataFrame({"standard_smiles": ["C", "CC"], "IC50": [4., 9.],
            "CC50": [7., 9.], "SI": [2., 1.], "IC50_qualifier": [">", ""],
            "CC50_qualifier": ["", ""], "target_is_label": [False, False]})
        reference = pd.DataFrame({"standard_smiles": ["C", "CCC"], "IC50": [np.nan, 1.],
            "CC50": [100., 1.], "SI": [10000., 1.], "target_is_label": [True, False]})
        result = restore_targets(frame, reference)
        self.assertTrue(np.isnan(result.IC50.iloc[0]))
        self.assertEqual(result.SI.iloc[0], 10000)
        self.assertEqual(result.IC50.iloc[1], 9)
        self.assertEqual(frame.IC50.iloc[0], 4)

    def test_score_preserves_original_rows_and_does_not_use_unmatched_targets(self):
        frame = pd.DataFrame({"standard_smiles": ["C"]})
        reference = pd.DataFrame({"standard_smiles": ["C", "C", "CC"], "SI": [1., 9., 999.]})
        truth, pred = original_pairs(frame, [3.], reference)
        np.testing.assert_array_equal(truth, [1., 9.])
        self.assertEqual(float(np.mean((truth-pred)**2)), 20)

    def test_weight_requires_fold_consistency_and_practical_gain(self):
        self.assertEqual(choose_weight([[100, 99, 101, 102, 103]]*3), 0)
        self.assertEqual(choose_weight([[100, 90, 100, 100, 100]]*3), .1)
        self.assertEqual(choose_weight([[100, 1, 101, 102, 103],
            [100, 101, 101, 102, 103], [100, 101, 101, 102, 103]]), 0)

    def test_three_fold_real_toy_run_uses_original_targets(self):
        smiles = ["c1ccccc1", "c1ccncc1", "C1CCCCC1", "c1ccoc1", "C1CCCC1", "c1ccsc1"]
        frame = pd.DataFrame({"standard_smiles": smiles, "IC50": np.arange(1, 7),
            "CC50": np.arange(1, 7)*10, "SI": 10., "source": "kaggle_train",
            "target_is_label": False, "IC50_qualifier": "", "CC50_qualifier": ""})
        reference = frame[["standard_smiles", "SI"]]
        result = select_on_train(frame, np.arange(24).reshape(6, 4), reference,
            {"n_estimators": 2, "max_depth": 1, "n_jobs": 1})
        self.assertEqual(result["weight"], 0)
        self.assertEqual(sum(fold["n_competition"] for fold in result["folds"]), 6)
        np.testing.assert_array_equal(classifier_labels(frame), np.zeros(6))


class CVWorkflow(_HybridWorkflow):
    def setUp(self):
        super().setUp()
        self.config["model"]["selection_protocol"] = "train_scaffold_cv"
        reference = self.frame[["standard_smiles", "IC50", "CC50", "SI"]].copy()
        reference["target_is_label"] = False
        reference.to_csv(self.root / "data/processed/kaggle_reference.csv", index=False)
        # Real three-fold fitting is tested above. Here isolate full workflow wiring.
        mock = patch("src.hybrid_cv.select_on_train", return_value={"weight": .1, "folds": []})
        mock.start()
        self.addCleanup(mock.stop)


del _HybridWorkflow
