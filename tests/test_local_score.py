import unittest
import json
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from src.local_score import score, paired_interval
from src.local_compare import fold_indices, benchmark
from src.local_compare import run
from src import xgb_workflow as workflow, splits
from src.data.standardize import standardize_smiles
from unittest.mock import patch


class Scores(unittest.TestCase):
    def test_raw_mse_includes_zero_targets_and_differs_from_log_metric(self):
        result = score([0, 10, 100], [1, 8, 80])
        self.assertAlmostEqual(result["mse"], 135)
        self.assertAlmostEqual(result["rmse"], np.sqrt(135))
        for truth, predicted in [([1], [np.nan]), ([1], [-1]), ([], []), ([1, 2], [1])]:
            with self.assertRaises(ValueError):
                score(truth, predicted)

    def test_bootstrap_does_not_treat_repeats_as_independent_observations(self):
        rows = pd.DataFrame({"original_row": [0, 1], "scaffold": ["a", "b"],
            "SI_true": [0., 0.], "variant1": [2., 2.], "variant3": [1., 1.]})
        once = paired_interval(rows, "variant3", draws=100)
        repeated = paired_interval(pd.concat([rows]*3, ignore_index=True), "variant3", draws=100)
        self.assertEqual(once, repeated)
        self.assertEqual(once["mean_mse_gain"], 3)

    def test_groups_and_fold_reproducibility(self):
        smiles = ["c1ccccc1", "Oc1ccccc1", "C1CCCCC1", "c1ccncc1", "C1CCCC1", "c1ccsc1"]
        result = fold_indices(smiles, "scaffold", 3, 123)
        self.assertEqual(result, fold_indices(smiles, "scaffold", 3, 123))
        self.assertTrue(any(0 in indices and 1 in indices for indices in result.values()))
        duplicate = fold_indices(["CCO", "CCO", "CCC", "CCCC"], "molecule", 2, 123)
        self.assertTrue(any(0 in indices and 1 in indices for indices in duplicate.values()))

    def test_benchmark_scores_same_original_rows_without_fit_predict_overlap(self):
        smiles = ["c1ccccc1", "C1CCCCC1", "c1ccncc1", "C1CCCC1", "c1ccsc1", "c1ccoc1"]
        frame = pd.DataFrame({"standard_smiles": smiles, "source": "kaggle_train",
            "IC50": np.arange(1, 7), "CC50": np.arange(1, 7)*10,
            "IC50_qualifier": "", "CC50_qualifier": ""})
        ref = pd.DataFrame({"original_row": np.arange(7), "standard_smiles": smiles+[smiles[0]],
            "SI": [10.]*7, "target_is_label": False})
        # Fake learners assert isolation at prediction time, rather than just trusting split metadata.
        class Learner:
            def __init__(self, *args):
                self.members = [self]
            def fit(self, x, ic, cc, sample_weight=None):
                self.keys = set(x[:, 0])
                if sample_weight is not None:
                    assert len(sample_weight) == len(x)
                return self
            def predict_si(self, x):
                assert not self.keys.intersection(set(x[:, 0]))
                return np.full(len(x), 10.)
        quality = pd.DataFrame({"standard_smiles": smiles, "public_weight": .5})
        with patch("src.local_compare.XGBEnsemble", Learner), patch("src.local_compare.XGBWeightedTwoHead", Learner):
            report, rows, _ = benchmark(frame, ref, quality, np.arange(24).reshape(6, 4), folds=3, repeats=2)
        self.assertEqual(report["original_competition_rows"], 7)
        self.assertFalse(report["holdout_used"])
        self.assertEqual(len(rows), 28)  # 7 rows x 2 repeats x 2 protocols
        for variants in report["protocols"].values():
            self.assertEqual(variants["variant3"]["mse"], 0)
            self.assertEqual(variants["variant1"]["n"], 14)

    def test_complete_runner_on_toy_data_excludes_holdout_and_writes_reports(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / "data/processed"
            directory.mkdir(parents=True)
            smiles = list(map(standardize_smiles, ["c1ccccc1", "C1CCCCC1", "c1ccncc1", "C1CCCC1",
                "c1ccsc1", "c1ccoc1", "N1CCCCC1", "O1CCNCC1"]))
            frame = pd.DataFrame({"standard_smiles": smiles, "source": "kaggle_train",
                "IC50": np.arange(1, 9, dtype=float), "CC50": np.arange(1, 9)*10., "SI": 10.,
                "IC50_qualifier": "", "CC50_qualifier": ""})
            frame.to_csv(directory / "dataset.csv", index=False)
            frame.iloc[:3].to_csv(directory / "train.csv", index=False)
            frame.iloc[3:6].to_csv(directory / "validation.csv", index=False)
            splits.save_manifest(directory / "splits.json",
                {"train": [0, 1, 2], "validation": [3, 4, 5], "holdout": [6, 7]}, 1, 8)
            reference = frame[["standard_smiles", "SI"]].copy()
            reference["original_row"] = np.arange(8)
            reference["target_is_label"] = False
            reference.to_csv(directory / "kaggle_reference.csv", index=False)
            pd.DataFrame({"standard_smiles": smiles, "public_weight": .5}).to_csv(directory / "public_quality.csv", index=False)
            config = {"seed": 123, "model": {"strategy": "xgb_weighted", "params":
                {"n_estimators": 2, "max_depth": 1, "n_jobs": 1, "min_child_weight": 1}},
                "local_comparison": {"folds": 3, "repeats": 1}}
            with patch.object(workflow, "ROOT", root):
                run(config)
            report = json.loads((root / "reports/local_comparison.json").read_text())
            self.assertEqual(report["original_competition_rows"], 6)
            self.assertFalse(report["holdout_used"])
            oof = pd.read_csv(root / "reports/local_comparison_oof.csv")
            self.assertTrue(set(oof.original_row).isdisjoint({6, 7}))
            self.assertEqual(len(oof), 12)
            self.assertFalse((root / "artifacts").exists())
