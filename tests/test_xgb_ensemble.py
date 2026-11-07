import json
import unittest
from unittest.mock import Mock
import numpy as np
import pandas as pd
from src import xgb_workflow as workflow
from src.models.xgb_two_head import XGBTwoHead
from src.models.xgb_ensemble import XGBEnsemble
from tests import test_xgb_two_head as fixtures


class Ensemble(unittest.TestCase):
    def setUp(self):
        fixtures.ToyWorkflow.setUp(self)
        self.config["training"] = {"mode": "train_validation"}
        self.config["model"]["strategy"] = "xgb_ensemble"
        self.config["submission"]["model_file"] = "artifacts/xgb_ensemble.joblib"
        self.config["submission"]["output"] = "reports/submission_xgb_ensemble.csv"

    def test_final_ensemble_serialization_members_and_submission(self):
        artifact = self.root / "artifacts/xgb_final.joblib"
        artifact.parent.mkdir()
        artifact.write_bytes(b"keep variant one")
        workflow.train(self.config)
        bundle = workflow.load_bundle(self.config)
        ensemble = bundle["model"]
        self.assertIs(type(ensemble), XGBEnsemble)
        self.assertEqual(len(ensemble.members), 5)
        self.assertEqual(ensemble.seeds, list(range(20261106, 20261111)))
        self.assertEqual(artifact.read_bytes(), b"keep variant one")
        for member, seed in zip(ensemble.members, ensemble.seeds):
            self.assertEqual(member.params["random_state"], seed)
            self.assertEqual(member.n_train["IC50"], 10)
            self.assertEqual(member.n_train["CC50"], 10)
        frame = self.frame.iloc[:10]
        matrix = workflow.features(frame)
        baseline = XGBTwoHead(self.config["model"]["params"]).fit(
            matrix, frame.IC50.to_numpy(), frame.CC50.to_numpy())
        np.testing.assert_allclose(ensemble.members[0].predict_si(matrix), baseline.predict_si(matrix))
        members = ensemble.predict_members_si(matrix)
        np.testing.assert_allclose(ensemble.predict_si(matrix), members.mean(axis=1))
        report = json.loads((self.root / "reports/xgb_ensemble_training.json").read_text())
        self.assertFalse(report["training"]["holdout_used"])
        self.assertFalse(report["training"]["independent_evaluation"])
        self.assertEqual(report["training"]["ensemble"]["members"], 5)
        pd.DataFrame({"index": [2, 1], "SMILES": ["CCO", "CCN"]}).to_csv(self.root / "test.csv", index=False)
        workflow.submission(self.config)
        result = pd.read_csv(self.root / "reports/submission_xgb_ensemble.csv")
        detail = pd.read_csv(self.root / "reports/submission_xgb_ensemble_members.csv")
        np.testing.assert_allclose(result["SI VACV"], detail.filter(like="SI_seed_20").mean(axis=1))
        self.assertEqual(result["index"].tolist(), [2, 1])
        self.assertTrue((detail.SI_seed_std >= 0).all())
        with self.assertRaisesRegex(ValueError, "independent validation"):
            workflow.evaluate(self.config)

    def test_unfitted_ensemble_is_rejected(self):
        with self.assertRaises(ValueError):
            XGBEnsemble().predict_si(np.ones((2, 4)))

    def test_averaging_is_arithmetic_in_raw_si_space(self):
        model = XGBEnsemble()
        values = np.array([[1., 4., 9., 16., 25.], [100., 10., 1., 5., 9.]])
        model.members = [Mock(predict_si=Mock(return_value=values[:, i])) for i in range(5)]
        actual = model.predict_si(np.ones((2, 4)))
        np.testing.assert_allclose(actual, values.mean(axis=1))
        self.assertFalse(np.allclose(actual, np.exp(np.log(values).mean(axis=1))))
