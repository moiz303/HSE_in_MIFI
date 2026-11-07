import json
import unittest
from unittest.mock import Mock, patch
import numpy as np
import pandas as pd
from src import xgb_workflow as workflow
from src.models.xgb_weighted_ensemble import XGBWeightedEnsemble
from tests import test_xgb_two_head as fixtures


class WeightedEnsemble(unittest.TestCase):
    def test_sample_weight_required(self):
        model = XGBWeightedEnsemble()
        x = np.ones((4, 4))
        with self.assertRaises(ValueError):
            model.fit(x, [1, 2, 3, 4], [10, 20, 30, 40], sample_weight=None)

    def test_unfitted_ensemble_is_rejected(self):
        with self.assertRaises(ValueError):
            XGBWeightedEnsemble().predict_si(np.ones((2, 4)))

    def test_averaging_is_arithmetic_in_raw_si_space(self):
        model = XGBWeightedEnsemble()
        values = np.array([[1., 4., 9., 16., 25.], [100., 10., 1., 5., 9.]])
        model.members = [Mock(predict_si=Mock(return_value=values[:, i])) for i in range(5)]
        actual = model.predict_si(np.ones((2, 4)))
        np.testing.assert_allclose(actual, values.mean(axis=1))
        self.assertFalse(np.allclose(actual, np.exp(np.log(values).mean(axis=1))))


class WeightedEnsembleFinal(unittest.TestCase):
    def setUp(self):
        fixtures.ToyWorkflow.setUp(self)
        self.config["training"] = {"mode": "train_validation"}
        self.config["model"]["strategy"] = "xgb_weighted_ensemble"
        self.config["submission"]["model_file"] = "artifacts/xgb_weighted_ensemble.joblib"
        self.config["submission"]["output"] = "reports/submission_xgb_weighted_ensemble.csv"
        frame = pd.read_csv(self.root / "data/processed/dataset.csv")
        pd.DataFrame({"standard_smiles": frame.standard_smiles, "public_weight": .5}).to_csv(
            self.root / "data/processed/public_quality.csv", index=False)
        for name in ("dataset.csv", "train.csv", "validation.csv"):
            path = self.root / "data/processed" / name
            rows = pd.read_csv(path)
            rows.loc[rows.standard_smiles.isin(["C", "CC", "CCC"]), "source"] = "chembl"
            rows.to_csv(path, index=False)

    def test_final_weighted_ensemble_pipeline(self):
        workflow.train(self.config)
        bundle = workflow.load_bundle(self.config)
        model = bundle["model"]
        self.assertIs(type(model), XGBWeightedEnsemble)
        self.assertEqual(len(model.members), 5)
        self.assertEqual(model.seeds, list(range(20261106, 20261111)))
        self.assertFalse(bundle["training"]["holdout_used"])
        self.assertFalse(bundle["training"]["independent_evaluation"])
        self.assertEqual(bundle["training"]["ensemble"]["members"], 5)
        self.assertIn("weights", bundle["training"])

        report = json.loads((self.root / "reports/xgb_weighted_ensemble_training.json").read_text())
        self.assertEqual(report["training"]["ensemble"]["members"], 5)
        self.assertFalse(report["training"]["holdout_used"])

        pd.DataFrame({"index": [1, 2], "SMILES": ["CCO", "CCN"]}).to_csv(self.root / "test.csv", index=False)
        workflow.submission(self.config)
        result = pd.read_csv(self.root / "reports/submission_xgb_weighted_ensemble.csv")
        detail = pd.read_csv(self.root / "reports/submission_xgb_weighted_ensemble_members.csv")
        self.assertEqual(len(result), 2)
        np.testing.assert_allclose(result["SI VACV"], detail.filter(like="SI_seed_20").mean(axis=1))
        self.assertTrue((detail.SI_seed_std >= 0).all())

        with self.assertRaisesRegex(ValueError, "independent validation"):
            workflow.evaluate(self.config)
