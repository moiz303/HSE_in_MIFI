"""Hybrid selection, portable serialization and full workflow on synthetic data."""
import tempfile
from pathlib import Path
from unittest.mock import patch
import joblib
import numpy as np
import pandas as pd
from src.models.xgb_hybrid import XGBHybrid, select_hybrid
from src import xgb_workflow as workflow
from tests.test_xgb_two_head import ToyWorkflow as _ToyWorkflow


class HybridWorkflow(_ToyWorkflow):
    def setUp(self):
        super().setUp()
        self.config["model"]["strategy"] = "xgb_hybrid"
        self.config["model"]["promote_candidate"] = True
        self.config["submission"]["model_file"] = "artifacts/xgb_hybrid.joblib"
        for filename in ("dataset.csv", "train.csv", "validation.csv"):
            path = self.root / "data/processed" / filename
            frame = pd.read_csv(path)
            frame["target_is_label"] = False
            frame.to_csv(path, index=False)

    def test_submission_requires_evaluation_and_selection_is_persisted(self):
        workflow.train(self.config)
        with self.assertRaisesRegex(ValueError, "evaluate"):
            workflow.submission(self.config)
        with patch.object(workflow, "select_hybrid", return_value=True):
            workflow.evaluate(self.config)
        bundle = workflow.load_bundle(self.config)
        self.assertTrue(bundle["model"].use_classifier)
        self.assertEqual(bundle["selection"]["selected"], "hybrid")
        with patch.object(workflow, "select_hybrid", return_value=False):
            workflow.evaluate(self.config)
        bundle = workflow.load_bundle(self.config)
        self.assertFalse(bundle["model"].use_classifier)
        self.assertEqual(bundle["selection"]["selected"], "baseline")

    def test_validation_win_does_not_promote_by_default(self):
        self.config["model"].pop("promote_candidate")
        workflow.train(self.config)
        with patch.object(workflow, "select_hybrid", return_value=True):
            workflow.evaluate(self.config)
        bundle = workflow.load_bundle(self.config)
        self.assertTrue(bundle["selection"]["eligible_on_validation"])
        self.assertFalse(bundle["model"].use_classifier)

    def test_hashes_ignore_platform_line_endings(self):
        path = self.root / "data/processed/train.csv"
        raw = path.read_bytes().replace(b"\r\n", b"\n")
        path.write_bytes(raw)
        expected = workflow.data_hashes()
        path.write_bytes(raw.replace(b"\n", b"\r\n"))
        self.assertEqual(expected, workflow.data_hashes())

    def test_selection_rule_and_native_classifier_roundtrip(self):
        self.assertTrue(select_hybrid(100, 97))
        self.assertFalse(select_hybrid(100, 99))
        self.assertFalse(select_hybrid(100, 110))
        self.assertFalse(select_hybrid(None, 1))
        x = np.arange(160, dtype=float).reshape(40, 4)
        model = XGBHybrid({"n_estimators": 3, "max_depth": 2, "n_jobs": 1})
        model.fit(x, np.arange(1, 41), np.arange(1, 41) * 10)
        labels = np.r_[np.zeros(20), np.ones(20)]
        model.fit_classifier(x, labels)
        base = model.predict_baseline_si(x)
        probability = model.predict_label_probability(x)
        np.testing.assert_allclose(model.predict_si(x), base)
        model.use_classifier = True
        expected = probability * 10000 + (1-probability) * base
        np.testing.assert_allclose(model.predict_si(x), expected)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.joblib"
            joblib.dump(model, path)
            restored = joblib.load(path)
            np.testing.assert_allclose(restored.predict_si(x), expected)
            self.assertTrue(restored.use_classifier)


del _ToyWorkflow
