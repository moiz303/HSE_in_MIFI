import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import joblib
import numpy as np
import pandas as pd
from src.models.xgb_analog import XGBAnalog, fingerprint_bits
from src.models.xgb_weighted import XGBWeightedTwoHead
from src.analog_compare import benchmark, choose_alpha, ALPHAS
from src.analog_compare import run as run_analog_comparison
from src import xgb_workflow as workflow
from src import splits
from src.data.standardize import standardize_smiles
from tests import test_final_training as final_fixtures


PARAMS = {"n_estimators": 2, "max_depth": 1, "n_jobs": 1, "min_child_weight": 1}


class Analogs(unittest.TestCase):
    def test_masks_similarity_accumulation_weighted_neighbors_and_portable_roundtrip(self):
        x = np.zeros((4, 2061), dtype=float)
        x[:2, :600] = 1  # overlap >255 catches uint8 dot-product overflow
        x[2, 600:1200] = 1
        x[3, 1200:1800] = 1
        ic, cc = [1., 4., np.nan, 16.], [10., np.nan, 40., 160.]
        weights = [1., .5, .25, 1.]
        model = XGBAnalog(PARAMS).fit(x, ic, cc, sample_weight=weights)
        baseline = XGBWeightedTwoHead(PARAMS).fit(x, ic, cc, sample_weight=weights)
        query = np.vstack([x[0], np.zeros(2061)])
        np.testing.assert_allclose(model.predict_si(query), baseline.predict_si(query))
        self.assertEqual(len(model.bank["IC50"]["log_values"]), 3)
        self.assertEqual(len(model.bank["CC50"]["log_values"]), 3)
        model.blend_weight = .25
        parts = model.predict_components(query)
        expected_analog = 10 / (4 ** (1/3))
        self.assertAlmostEqual(parts["SI_analog"][0], expected_analog)
        self.assertAlmostEqual(parts["IC50_neighbor_similarity"][0], 1)
        self.assertEqual(parts["IC50_neighbors"][0], 2)
        self.assertEqual(parts["CC50_neighbors"][0], 1)
        np.testing.assert_array_equal(parts["analog_available"], [True, False])
        self.assertEqual(parts["SI_selected"][1], parts["SI_baseline"][1])
        np.testing.assert_allclose(parts["SI_selected"], .75*parts["SI_baseline"]+.25*parts["SI_analog"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"model.joblib"
            joblib.dump(model, path)
            restored = joblib.load(path)
            np.testing.assert_allclose(restored.predict_si(query), model.predict_si(query))
        with self.assertRaises(ValueError):
            fingerprint_bits(np.ones((2, 10)))
        with self.assertRaises(ValueError):
            fingerprint_bits(np.full((2, 2061), 2))

    def test_selection_requires_main_gain_and_respects_stress_and_measured_guards(self):
        protocols = {}
        for protocol in ("molecule", "scaffold"):
            protocols[protocol] = {candidate: {"mse": 110., "repeat_mse": [110.]*3,
                "measured": {"mse": 110.}} for candidate in ALPHAS}
            protocols[protocol]["weighted"] = {"mse": 100., "repeat_mse": [100.]*3, "measured": {"mse": 100.}}
            protocols[protocol]["alpha_025"] = {"mse": 90., "repeat_mse": [90.]*3, "measured": {"mse": 90.}}
        self.assertEqual(choose_alpha(protocols)["alpha"], .25)
        bad = copy.deepcopy(protocols)
        bad["scaffold"]["alpha_025"]["mse"] = 103.
        self.assertEqual(choose_alpha(bad)["alpha"], 0)
        bad = copy.deepcopy(protocols)
        bad["molecule"]["alpha_025"]["measured"]["mse"] = 103.
        self.assertEqual(choose_alpha(bad)["alpha"], 0)

    def test_oof_neighbor_bank_never_contains_validation_queries(self):
        smiles = ["c1ccccc1", "C1CCCCC1", "c1ccncc1", "C1CCCC1", "c1ccsc1", "c1ccoc1"]
        frame = pd.DataFrame({"standard_smiles": smiles, "source": "kaggle_train",
            "IC50": np.arange(1, 7), "CC50": np.arange(1, 7)*10,
            "IC50_qualifier": "", "CC50_qualifier": ""})
        reference = pd.DataFrame({"standard_smiles": smiles, "SI": 10., "original_row": np.arange(6), "target_is_label": False})
        quality = pd.DataFrame({"standard_smiles": smiles, "public_weight": .5})
        class FoldModel:
            def __init__(self, params):
                pass
            def fit(self, x, ic, cc, sample_weight):
                self.bank = set(x[:, 0])
                return self
            def predict_components(self, x):
                assert not self.bank.intersection(set(x[:, 0]))
                return {"SI_baseline": np.full(len(x), 10.), "SI_analog": np.full(len(x), 20.),
                    "analog_available": np.ones(len(x), dtype=bool)}
        with patch("src.analog_compare.XGBAnalog", FoldModel):
            report, oof, _ = benchmark(frame, reference, quality, np.arange(24).reshape(6, 4), folds=3, repeats=2)
        self.assertEqual(len(oof), 24)
        self.assertEqual(report["selection"]["alpha"], 0)
        self.assertFalse(report["holdout_used"])

    def test_real_toy_oof_runner_writes_selection_and_reports(self):
        smiles = list(map(standardize_smiles, ["c1ccccc1", "C1CCCCC1", "c1ccncc1", "C1CCCC1", "c1ccsc1", "c1ccoc1"]))
        frame = pd.DataFrame({"standard_smiles": smiles, "source": "kaggle_train",
            "IC50": np.arange(1, 7, dtype=float), "CC50": np.arange(1, 7)*10.,
            "IC50_qualifier": "", "CC50_qualifier": ""})
        quality = pd.DataFrame({"standard_smiles": smiles, "public_weight": .5})
        reference = pd.DataFrame({"standard_smiles": smiles, "SI": 10., "original_row": np.arange(6), "target_is_label": False})
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / "data/processed"
            directory.mkdir(parents=True)
            frame.iloc[:3].to_csv(directory / "train.csv", index=False)
            frame.iloc[3:].to_csv(directory / "validation.csv", index=False)
            splits.save_manifest(directory / "splits.json", {"train": [0, 1, 2], "validation": [3, 4, 5], "holdout": []}, 123, 6)
            quality.to_csv(directory / "public_quality.csv", index=False)
            reference.to_csv(directory / "kaggle_reference.csv", index=False)
            config = {"seed": 123, "model": {"strategy": "xgb_analog", "params": PARAMS},
                "analog_comparison": {"folds": 3, "repeats": 1}}
            with patch.object(workflow, "ROOT", root):
                result = run_analog_comparison(config, frame, workflow.features(frame), quality)
            self.assertEqual(result["original_competition_rows"], 6)
            self.assertIn(result["selection"]["alpha"], [0, .1, .25, .5])
            self.assertEqual(len(pd.read_csv(root / "reports/analog_comparison_oof.csv")), 12)
            self.assertTrue((root / "reports/analog_comparison.json").exists())
            self.assertFalse((root / "artifacts").exists())


class AnalogFinal(final_fixtures.FinalTraining):
    def setUp(self):
        super().setUp()
        self.config["model"]["strategy"] = "xgb_analog"
        self.config["submission"]["model_file"] = "artifacts/xgb_analog.joblib"
        pd.DataFrame({"standard_smiles": self.frame.standard_smiles, "public_weight": .5}).to_csv(
            self.root / "data/processed/public_quality.csv", index=False)
        mock = patch("src.analog_compare.run", return_value={"selection": {"alpha": .25, "selected": "alpha_025"}})
        mock.start()
        self.addCleanup(mock.stop)

    def test_refit_uses_ten_development_rows_and_excludes_two_holdout_rows(self):
        workflow.train(self.config)
        bundle = workflow.load_bundle(self.config)
        model = bundle["model"]
        self.assertEqual(bundle["training"]["rows"], 10)
        self.assertFalse(bundle["training"]["holdout_used"])
        self.assertEqual(model.blend_weight, .25)
        self.assertEqual(len(model.bank["IC50"]["log_values"]), 10)
        self.assertTrue((self.root / "reports/xgb_analog_training.json").exists())
        test = pd.DataFrame({"index": [2, 1], "SMILES": ["CCO", "CCN"],
            "IC50_uM_VACV": [1, 1], "CC50_uM": [1, 1]})
        path = self.root / "test.csv"
        test.to_csv(path, index=False)
        workflow.submission(self.config)
        first = pd.read_csv(self.root / "reports/submission.csv")
        test["IC50_uM_VACV"] = [np.nan, 1e-12]
        test["CC50_uM"] = [1e9, np.nan]
        test.to_csv(path, index=False)
        workflow.submission(self.config)
        pd.testing.assert_frame_equal(first, pd.read_csv(self.root / "reports/submission.csv"))
        self.assertTrue((self.root / "reports/xgb_analog_neighbors.csv").exists())
