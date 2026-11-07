"""Final refit must include validation, exclude holdout, and never claim validation quality."""
import json
import unittest
from unittest.mock import patch
import pandas as pd
from src import xgb_workflow as workflow, pipeline
from legacy import train_model_v1 as train_model
from tests import test_xgb_two_head as fixtures


class FinalTraining(unittest.TestCase):
    def setUp(self):
        fixtures.ToyWorkflow.setUp(self)
        self.config["training"] = {"mode": "train_validation"}
        self.config["submission"]["model_file"] = "artifacts/xgb_final.joblib"

    def test_refit_uses_ten_development_rows_and_excludes_two_holdout_rows(self):
        artifacts = self.root / "artifacts"
        artifacts.mkdir()
        baseline = artifacts / "xgb_baseline.joblib"
        baseline.write_bytes(b"preserve verified baseline")
        workflow.train(self.config)
        bundle = workflow.load_bundle(self.config)
        self.assertEqual(bundle["training"]["parts"], ["train", "validation"])
        self.assertEqual(bundle["training"]["rows"], 10)
        self.assertEqual(bundle["model"].n_train["IC50"], 10)
        self.assertEqual(bundle["model"].n_train["CC50"], 10)
        self.assertFalse(bundle["training"]["holdout_used"])
        self.assertEqual(baseline.read_bytes(), b"preserve verified baseline")
        report = json.loads((self.root / "reports/xgb_final_training.json").read_text())
        self.assertFalse(report["training"]["independent_evaluation"])
        self.assertNotIn("validation", report)
        pd.DataFrame({"index": [5], "SMILES": ["CCO"]}).to_csv(self.root / "test.csv", index=False)
        workflow.submission(self.config)
        self.assertEqual(len(pd.read_csv(self.root / "reports/submission.csv")), 1)

    def test_final_validation_scoring_and_invalid_modes_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "independent validation"):
            workflow.evaluate(self.config)
        self.config["training"]["mode"] = "all_including_holdout"
        with self.assertRaises(ValueError):
            workflow.validate_config(self.config)
        self.config["training"]["mode"] = "train_validation"
        self.config["model"]["strategy"] = "xgb_hybrid"
        with self.assertRaises(ValueError):
            workflow.validate_config(self.config)

    def test_both_entrypoints_skip_validation_scoring(self):
        with patch.object(train_model, "load_config", return_value=self.config), \
                patch.object(workflow, "train") as fit, patch.object(workflow, "evaluate") as evaluate:
            train_model.main([])
            fit.assert_called_once_with(self.config)
            evaluate.assert_not_called()
        calls = []
        steps = {name: (lambda config, name=name: calls.append(name)) for name in pipeline.STEPS}
        with patch.object(pipeline, "load_config", return_value=self.config), patch.dict(pipeline.STEPS, steps):
            pipeline.main(["all"])
        self.assertEqual(calls, ["train", "submission"])
