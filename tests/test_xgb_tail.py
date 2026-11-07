import copy
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from src.models.xgb_tail import tail_weights, PROFILES
from src.models.xgb_two_head import XGBTwoHead
from src.tail_compare import benchmark, choose_profile
from src import xgb_workflow as workflow
from tests import test_final_training as fixtures


class TailWeights(unittest.TestCase):
    def test_only_exact_known_measured_pairs_are_boosted_and_capped(self):
        frame = pd.DataFrame({"IC50": [1]*8, "CC50": [100, 400, 1e6, 1e6, 1e6, np.nan, 1e6, 1e6],
            "IC50_qualifier": [""]*6 + [">"] + [""], "CC50_qualifier": [""]*7 + ["<"],
            "target_is_label": [False, False, False, True, pd.NA, False, False, False],
            "SI": [10000]*8})
        base = np.full(8, .5)
        weights = tail_weights(frame, base, "ic4")
        np.testing.assert_allclose(weights["IC50"], base * [1, 2, 4, 1, 1, 1, 1, 1])
        np.testing.assert_array_equal(weights["CC50"], base)
        both = tail_weights(frame, base, "both2")
        np.testing.assert_allclose(both["CC50"], base * [1, 2, 2, 1, 1, 1, 1, 1])
        for vector in tail_weights(frame, base, "weighted").values():
            np.testing.assert_array_equal(vector, base)

    def test_head_specific_weights_are_masked_normalized_and_validated_before_fit(self):
        seen = []
        class Head:
            def fit(self, x, y, sample_weight):
                seen.append(sample_weight)
        weights = {"IC50": [1, 2, 4], "CC50": [1, 1, 1]}
        x = np.arange(9).reshape(3, 3)
        with patch.object(XGBTwoHead, "_build_head", return_value=Head()):
            XGBTwoHead().fit(x, [1, np.nan, 1], [1, 1, 1], sample_weight=weights)
        np.testing.assert_allclose(seen[0], [.4, 1.6])
        np.testing.assert_allclose(seen[1], [1, 1, 1])
        for bad in ({"IC50": [1]*3}, {"IC50": [1]*3, "CC50": [1, 0, 1]}):
            with self.assertRaises(ValueError):
                XGBTwoHead().fit(x, [1]*3, [1]*3, sample_weight=bad)

    def test_selection_rejects_ordinary_value_damage_and_unstable_wins(self):
        protocols = {p: {c: {"mse": 100., "repeat_mse": [100.]*3,
            "measured": {"mse": 100.}, "low_si": {"mse": 100.}} for c in PROFILES}
            for p in ("molecule", "scaffold")}
        for p in protocols.values():
            p["ic2"].update(mse=90., repeat_mse=[90.]*3)
        self.assertEqual(choose_profile(protocols)["selected"], "ic2")
        bad = copy.deepcopy(protocols)
        bad["molecule"]["ic2"]["low_si"]["mse"] = 103.
        self.assertEqual(choose_profile(bad)["selected"], "weighted")
        bad = copy.deepcopy(protocols)
        bad["scaffold"]["ic2"]["mse"] = 103.
        self.assertEqual(choose_profile(bad)["selected"], "weighted")
        bad = copy.deepcopy(protocols)
        bad["molecule"]["ic2"]["repeat_mse"] = [70., 101., 101.]
        self.assertEqual(choose_profile(bad)["selected"], "weighted")

    def test_oof_models_and_target_dependent_weights_only_see_fit_rows(self):
        smiles = ["c1ccccc1", "C1CCCCC1", "c1ccncc1", "C1CCCC1", "c1ccsc1", "c1ccoc1"]
        frame = pd.DataFrame({"standard_smiles": smiles, "source": "kaggle_train",
            "IC50": np.arange(1, 7), "CC50": np.arange(1, 7)*1000,
            "IC50_qualifier": "", "CC50_qualifier": "", "target_is_label": False})
        reference = pd.DataFrame({"standard_smiles": smiles, "SI": 1000., "original_row": np.arange(6), "target_is_label": False})
        quality = pd.DataFrame({"standard_smiles": smiles, "public_weight": .5})
        calls = []
        class FoldModel:
            def __init__(self, params):
                pass
            def fit(self, x, ic, cc, sample_weight):
                self.fit_rows = set(x[:, 0])
                self.asserted = np.array_equal(ic, x[:, 0]+1)
                assert self.asserted
                calls.append(self.fit_rows)
                return self
            def predict_si(self, x):
                assert not self.fit_rows.intersection(set(x[:, 0]))
                return np.full(len(x), 1000.)
        original_weights = tail_weights
        def checked_weights(part, reliability, profile):
            assert len(part) == 4  # no validation targets supplied to weight computation
            return original_weights(part, reliability, profile)
        matrix = np.arange(6).reshape(6, 1)
        with patch("src.tail_compare.XGBTail", FoldModel), patch("src.tail_compare.tail_weights", checked_weights):
            report, oof, _ = benchmark(frame, reference, quality, matrix, folds=3, repeats=2)
        self.assertEqual(len(calls), 48)
        self.assertEqual(len(oof), 24)
        self.assertEqual(report["selection"]["selected"], "weighted")


class TailFinal(fixtures.FinalTraining):
    def setUp(self):
        super().setUp()
        self.config["model"]["strategy"] = "xgb_tail"
        self.config["submission"]["model_file"] = "artifacts/xgb_tail.joblib"
        for name in ("dataset.csv", "train.csv", "validation.csv"):
            path = self.root / "data/processed" / name
            frame = pd.read_csv(path)
            frame["target_is_label"] = False
            frame.to_csv(path, index=False)
        pd.DataFrame({"standard_smiles": self.frame.standard_smiles, "public_weight": .5}).to_csv(
            self.root / "data/processed/public_quality.csv", index=False)
        mock = patch("src.tail_compare.run", return_value={"selection": {"selected": "ic2"}})
        mock.start()
        self.addCleanup(mock.stop)

    def test_refit_uses_ten_development_rows_and_excludes_two_holdout_rows(self):
        workflow.train(self.config)
        bundle = workflow.load_bundle(self.config)
        self.assertEqual(bundle["training"]["rows"], 10)
        self.assertFalse(bundle["training"]["holdout_used"])
        self.assertEqual(bundle["model"].profile, "ic2")
        self.assertTrue((self.root / "reports/xgb_tail_training.json").exists())
        frame = pd.DataFrame({"index": [1, 2], "SMILES": ["CCO", "CCN"],
            "CC50_uM": [1., 2.], "IC50_uM_VACV": [3., 4.]})
        frame.to_csv(self.root / "test.csv", index=False)
        workflow.submission(self.config)
        first = pd.read_csv(self.root / "reports/submission.csv")
        frame["CC50_uM"] = [1e9, np.nan]
        frame["IC50_uM_VACV"] = [np.nan, 1e-12]
        frame.to_csv(self.root / "test.csv", index=False)
        workflow.submission(self.config)
        pd.testing.assert_frame_equal(first, pd.read_csv(self.root / "reports/submission.csv"))
