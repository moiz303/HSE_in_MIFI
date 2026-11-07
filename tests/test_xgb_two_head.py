"""Проверка нового пути только на искусственных данных, без реального обучения."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd

from src import splits, xgb_workflow as workflow
from src.models.xgb_two_head import XGBTwoHead, exact_target


PARAMS = {"n_estimators": 4, "max_depth": 2, "n_jobs": 1, "min_child_weight": 1}


class Heads(unittest.TestCase):
    def test_independent_missing_targets_and_ratio(self):
        features = np.arange(48, dtype=float).reshape(12, 4)
        ic50 = np.arange(1, 13, dtype=float)
        cc50 = 10 * ic50.copy()
        ic50[:2] = np.nan
        cc50[-3:] = np.nan
        model = XGBTwoHead(PARAMS).fit(features, ic50, cc50)
        self.assertEqual(model.n_train["IC50"], 10)
        self.assertEqual(model.n_train["CC50"], 9)
        logs = model.predict_log_concentrations(features)
        np.testing.assert_allclose(model.predict_si(features), 10.0 ** (logs["CC50"] - logs["IC50"]))

    def test_censoring_is_per_head_and_si_labels_do_not_invent_targets(self):
        frame = pd.DataFrame({"IC50": [1, 2, np.nan, 0, np.inf],
            "CC50": [10, 20, 30, 40, 50], "IC50_qualifier": [">", "", "", "", ""],
            "CC50_qualifier": ["", "<;>", "", "", ""], "SI": [1, 10, 10000, 1, 1]})
        np.testing.assert_array_equal(np.isfinite(exact_target(frame, "IC50")), [False, True, False, False, False])
        np.testing.assert_array_equal(np.isfinite(exact_target(frame, "CC50")), [True, False, True, True, True])
        with self.assertRaisesRegex(ValueError, "цензуру"):
            exact_target(frame.drop(columns="IC50_qualifier"), "IC50")

    def test_bad_parameters_and_empty_head_fail(self):
        with self.assertRaises(ValueError):
            XGBTwoHead({"max_iter": 100})
        with self.assertRaisesRegex(ValueError, "IC50"):
            XGBTwoHead(PARAMS).fit(np.ones((3, 2)), [np.nan] * 3, [1, 2, 3])
        with self.assertRaises(ValueError):
            XGBTwoHead(clip_si=0)

    def test_standardization_matches_salts_and_rejects_garbage(self):
        matrix = workflow.features(pd.DataFrame({"SMILES": ["CCO", "CCO.[Na+]"]}), "SMILES")
        np.testing.assert_array_equal(matrix[0], matrix[1])
        with self.assertRaisesRegex(ValueError, "SMILES"):
            workflow.features(pd.DataFrame({"SMILES": ["garbage"]}), "SMILES")


class ToyWorkflow(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        patcher = patch.object(workflow, "ROOT", self.root)
        patcher.start()
        self.addCleanup(patcher.stop)
        directory = self.root / "data/processed"
        directory.mkdir(parents=True)
        smiles = ["C", "CC", "CCC", "CCCC", "CO", "CCO", "CCCO", "CN", "CCN", "CCCN", "CS", "CCS"]
        ic50 = np.arange(1, 13, dtype=float)
        self.frame = pd.DataFrame({"standard_smiles": smiles, "IC50": ic50,
            "CC50": 10 * ic50, "SI": np.full(12, 10.0), "source": "kaggle_train",
            "IC50_qualifier": "", "CC50_qualifier": ""})
        self.frame.to_csv(directory / "dataset.csv", index=False)
        split = {"train": list(range(8)), "validation": [8, 9], "holdout": [10, 11]}
        splits.save_manifest(directory / "splits.json", split, 1, 12)
        for part in ("train", "validation"):
            self.frame.iloc[split[part]].to_csv(directory / f"{part}.csv", index=False)
        self.config = {"model": {"strategy": "xgb_two_head", "params": PARAMS, "clip_si": None},
            "features": {"morgan": True, "descriptors": True},
            "submission": {"model_file": "artifacts/xgb_two_head.joblib", "test_csv": "test.csv",
                "smiles_column": "SMILES", "id_column": "index", "target_column": "SI VACV",
                "output": "reports/submission.csv"}}

    def test_train_evaluate_submission_roundtrip_and_no_test_target_leak(self):
        workflow.train(self.config)
        workflow.evaluate(self.config)
        report = json.loads((self.root / "reports/xgb_metrics.json").read_text())
        self.assertNotIn("holdout", report)
        self.assertEqual(report["validation"]["SI_competition"]["n"], 2)
        self.assertEqual(report["n_train"]["IC50"], 8)
        frame = pd.DataFrame({"index": [8, 2], "SMILES": ["CCO", "CCN"], "CC50_uM": [1, 2], "IC50_uM_VACV": [1, 2]})
        frame.to_csv(self.root / "test.csv", index=False)
        workflow.submission(self.config)
        first = pd.read_csv(self.root / "reports/submission.csv")
        frame["CC50_uM"] = [1e12, np.nan]
        frame["IC50_uM_VACV"] = [np.nan, 1e-12]
        frame.to_csv(self.root / "test.csv", index=False)
        workflow.submission(self.config)
        pd.testing.assert_frame_equal(first, pd.read_csv(self.root / "reports/submission.csv"))
        self.assertEqual(first["index"].tolist(), [8, 2])
        bundle = joblib.load(self.root / "artifacts/xgb_two_head.joblib")
        self.assertEqual(bundle["constant_si"], 10.0)

    def test_modified_export_and_overlap_fail_before_training(self):
        path = self.root / "data/processed/validation.csv"
        frame = pd.read_csv(path)
        frame.loc[0, "standard_smiles"] = "C"
        frame.to_csv(path, index=False)
        with self.assertRaises(AssertionError):
            workflow.load_parts()

    def test_changed_config_or_data_rejects_artifact(self):
        workflow.train(self.config)
        modified = {**self.config, "features": {"morgan": False, "descriptors": True}}
        with self.assertRaises(ValueError):
            workflow.load_bundle(modified)
        path = self.root / "data/processed/train.csv"
        path.write_text(path.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "изменились"):
            workflow.evaluate(self.config)


if __name__ == "__main__":
    unittest.main()
