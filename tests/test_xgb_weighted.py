import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from src.data.quality_weights import build_quality, row_weights
from src.models.xgb_two_head import XGBTwoHead
from src.models.xgb_weighted import XGBWeightedTwoHead
from src import xgb_workflow as workflow
from tests import test_final_training as final_fixtures


class Weights(unittest.TestCase):
    def test_worst_pairing_is_used_and_competition_rows_keep_full_weight(self):
        raw = pd.DataFrame({"smiles": ["CCO", "CCO", "CCC"],
            "cc50_cell_line_match": ["vaccinia_same_cell_line", "any_in_document", "same_cell_line"]})
        quality = build_quality(raw)
        frame = pd.DataFrame({"standard_smiles": ["CCO", "CCC", "CCC"],
            "source": ["chembl", "kaggle_train", "chembl;kaggle_train"]})
        np.testing.assert_allclose(row_weights(frame, quality), [.125, 1, 1])
        frame.loc[0, "standard_smiles"] = "CCCC"
        with self.assertRaises(ValueError):
            row_weights(frame, quality)

    def test_each_head_receives_its_masked_normalized_weights(self):
        seen = []
        class Head:
            def fit(self, x, y, sample_weight):
                seen.append(sample_weight)
        x = np.arange(16).reshape(4, 4)
        with patch.object(XGBWeightedTwoHead, "_build_head", return_value=Head()):
            model = XGBWeightedTwoHead().fit(x, [1, np.nan, 3, 4], [np.nan, 2, 3, 4],
                sample_weight=[1, .5, .25, .125])
        first, second = np.array([1, .25, .125]), np.array([.5, .25, .125])
        np.testing.assert_allclose(seen[0], first/first.mean())
        np.testing.assert_allclose(seen[1], second/second.mean())
        self.assertEqual(model.n_train["IC50"], 3)
        for weights in ([1, 0, 1, 1], [1, np.nan, 1, 1], [1]):
            with self.assertRaises(ValueError):
                XGBWeightedTwoHead().fit(x, [1]*4, [1]*4, sample_weight=weights)


class WeightedFinal(final_fixtures.FinalTraining):
    def setUp(self):
        super().setUp()
        self.config["model"]["strategy"] = "xgb_weighted"
        self.config["submission"]["model_file"] = "artifacts/xgb_weighted.joblib"
        frame = pd.read_csv(self.root / "data/processed/dataset.csv")
        pd.DataFrame({"standard_smiles": frame.standard_smiles, "public_weight": .5}).to_csv(
            self.root / "data/processed/public_quality.csv", index=False)
        for name in ("dataset.csv", "train.csv", "validation.csv"):
            path = self.root / "data/processed" / name
            rows = pd.read_csv(path)
            rows.loc[rows.standard_smiles.isin(["C", "CC", "CCC"]), "source"] = "chembl"
            rows.to_csv(path, index=False)

    def test_refit_uses_ten_development_rows_and_excludes_two_holdout_rows(self):
        workflow.train(self.config)
        bundle = workflow.load_bundle(self.config)
        self.assertEqual(bundle["training"]["rows"], 10)
        self.assertEqual(bundle["model"].n_train["IC50"], 10)
        self.assertFalse(bundle["training"]["holdout_used"])
        self.assertLess(bundle["model"].weight_summary["IC50"]["min"],
                        bundle["model"].weight_summary["IC50"]["max"])
        self.assertTrue((self.root / "reports/xgb_weighted_training.json").exists())
        frame = pd.DataFrame({"index": [1], "SMILES": ["CCO"]})
        frame.to_csv(self.root / "test.csv", index=False)
        workflow.submission(self.config)
        self.assertEqual(len(pd.read_csv(self.root / "reports/submission.csv")), 1)
