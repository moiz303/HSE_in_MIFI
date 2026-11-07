"""Проверки D на синтетических данных и двух различимых моделях."""
import json
import tempfile
import unittest
from functools import partial
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd

from src import evaluate, pipeline, splits, run_record
from src.run_record import checksum


class ConstantModel:
    def __init__(self, log_si):
        self.log_si = log_si

    def predict_log_si(self, features):
        return np.full(len(features), self.log_si)


class Delivery(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        for name, relative in {
            "ROOT": ".", "PROCESSED": "data/processed", "DATASET": "data/processed/dataset.csv",
            "MANIFEST": "data/processed/splits.json", "REPORTS": "reports", "ARTIFACTS": "artifacts",
            "DEFAULT_CONFIG": "configs/baseline.json",
        }.items():
            self.enterContext(patch.object(pipeline, name, self.root / relative))
        self.enterContext(patch.object(run_record, "ROOT", self.root))
        # Интервалы те же, но на небольшом числе повторов для быстрого теста проводки.
        self.enterContext(patch.object(evaluate, "bootstrap_ci",
                                       partial(evaluate.bootstrap_ci, n_resamples=20)))
        pipeline.PROCESSED.mkdir(parents=True)
        pipeline.ARTIFACTS.mkdir()
        data = pd.DataFrame({"standard_smiles": ["CCO", "c1ccccc1", "C1CCCCC1"],
                                  "SI": [10., 20., 30.], "source": ["chembl", "kaggle_train", "chembl"]})
        data.to_csv(pipeline.DATASET, index=False)
        data.iloc[:1].to_csv(pipeline.PROCESSED / "train.csv", index=False)
        splits.save_manifest(pipeline.MANIFEST, {"train": [0], "validation": [1], "holdout": [2]}, 7, 3)
        self.config = {"seed": 7, "features": {"morgan": True, "descriptors": False},
                       "model": {"strategy": "direct", "params": {"max_iter": 2}},
                       "splits": {"seed": 7, "fractions": {"train": .4, "validation": .3, "holdout": .3}},
                       "submission": {"model_file": "artifacts/selected.joblib", "test_csv": "test.csv",
                                      "smiles_column": "SMILES", "id_column": "index",
                                      "target_column": "SI VACV", "output": "reports/submission.csv"}}
        for name, value in (("model.joblib", 1.), ("selected.joblib", 2.)):
            joblib.dump({"model": ConstantModel(value), "train_smiles": ["CCO"]}, pipeline.ARTIFACTS / name)
        pd.DataFrame({"index": [5, 7], "SMILES": ["CCO", "invalid"]}).to_csv(self.root / "test.csv", index=False)
        pipeline.DEFAULT_CONFIG.parent.mkdir()
        pipeline.DEFAULT_CONFIG.write_text(json.dumps(self.config))
        (self.root / "requirements.txt").write_text("numpy==test\n")

    def read(self, name):
        return json.loads((pipeline.REPORTS / name).read_text())

    def build(self):
        pipeline.step_evaluate(self.config)
        pipeline.step_submission(self.config)

    def test_evaluation_and_submission_use_explicit_model(self):
        self.build()
        metrics = self.read("metrics.json")
        metadata = self.read("submission.json")
        self.assertEqual(metrics["model_sha256"], metadata["model_sha256"])
        self.assertAlmostEqual(metrics["validation"]["competition_mse"], (100-20)**2)
        self.assertEqual(len(metrics["validation"]["competition_mse_ci95"]), 2)
        self.assertAlmostEqual(metrics["validation"]["kaggle_train"]["competition_mse"], (100-20)**2)
        np.testing.assert_allclose(pd.read_csv(pipeline.REPORTS / "submission.csv")["SI VACV"], [100, 100])
        domain = pd.read_csv(pipeline.REPORTS / "applicability_submission.csv")
        self.assertEqual(domain["id"].tolist(), [5, 7])
        self.assertEqual(domain["outside_domain"].tolist(), [False, True])
        self.assertEqual(domain["max_tanimoto"].tolist(), [1., 0.])

    def test_fallback_is_model_joblib_not_strategy_filename(self):
        del self.config["submission"]["model_file"]
        self.assertEqual(pipeline.model_path(self.config), pipeline.ARTIFACTS / "model.joblib")

    def test_release_tracks_artifacts_and_rejects_stale_results(self):
        self.build()
        pipeline.step_release(self.config)
        release = self.read("release.json")
        for name in ("artifacts/selected.joblib", "data/processed/splits.json", "reports/submission.csv"):
            self.assertEqual(release["sha256"][name], checksum(self.root / name))
        joblib.dump({"model": ConstantModel(3)}, pipeline.ARTIFACTS / "selected.joblib")
        with self.assertRaisesRegex(SystemExit, "другой модели"):
            pipeline.step_release(self.config)

    def test_release_checks_holdout_provenance(self):
        self.build()
        pipeline.step_holdout(self.config)
        pipeline.step_release(self.config)
        self.assertIn("reports/holdout.json", self.read("release.json")["sha256"])
        held = self.read("holdout.json")
        held["model_sha256"] = "different"
        (pipeline.REPORTS / "holdout.json").write_text(json.dumps(held))
        with self.assertRaisesRegex(SystemExit, "holdout"):
            pipeline.step_release(self.config)

    def test_release_rejects_edited_submission(self):
        self.build()
        (pipeline.REPORTS / "submission.csv").write_text("changed")
        with self.assertRaisesRegex(SystemExit, "CSV изменился"):
            pipeline.step_release(self.config)

    def test_holdout_is_separate_and_cannot_repeat(self):
        self.build()
        self.assertFalse((pipeline.REPORTS / "holdout.json").exists())
        pipeline.step_holdout(self.config)
        result = self.read("holdout.json")
        self.assertEqual(result["status"], "success")
        self.assertIn("holdout", result)
        self.assertNotIn("validation", result)
        with self.assertRaisesRegex(SystemExit, "уже запускался"):
            pipeline.step_holdout(self.config)

    def test_holdout_failure_also_blocks_repeat(self):
        with patch.object(pipeline, "evaluate_parts", side_effect=ValueError("broken")):
            with self.assertRaises(ValueError):
                pipeline.step_holdout(self.config)
        self.assertEqual(self.read("holdout.json")["status"], "failed")
        with self.assertRaises(SystemExit):
            pipeline.step_holdout(self.config)

    def test_holdout_rejects_training_overlap(self):
        joblib.dump({"model": ConstantModel(2), "train_smiles": ["C1CCCCC1"]},
                    pipeline.ARTIFACTS / "selected.joblib")
        with self.assertRaisesRegex(AssertionError, "молекула"):
            pipeline.step_holdout(self.config)

    def test_all_uses_new_model_and_creates_release(self):
        pipeline.main(["all"])
        metadata = self.read("submission.json")
        self.assertEqual(metadata["model_file"], "artifacts/model.joblib")
        self.assertEqual(metadata["model_sha256"], self.read("metrics.json")["model_sha256"])
        self.assertEqual(metadata["model_sha256"], self.read("release.json")["model_sha256"])
        self.assertFalse((pipeline.REPORTS / "holdout.json").exists())

    def test_mse_interval_is_deterministic_and_contains_point_estimate(self):
        true = np.array([0., 1., 2., 1.5])
        pred = true + [.1, -.2, .3, -.1]
        result = evaluate.report(true, pred)
        low, high = result["competition_mse_ci95"]
        self.assertLessEqual(low, result["competition_mse"])
        self.assertGreaterEqual(high, result["competition_mse"])
        self.assertEqual(result, evaluate.report(true, pred))
