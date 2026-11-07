import importlib.util
import tempfile
from pathlib import Path
import unittest
import hashlib
from unittest.mock import patch
import joblib
import numpy as np
import pandas as pd
from src.models.architecture_heads import FrozenHeads, target_scaler
from src.models.joint_gp import JointGP, tanimoto
from src.score_audit import official_reference_mse


class Scores(unittest.TestCase):
    def test_checkpoint_transfer_resumes_prefix_and_verifies_publisher_hash(self):
        from src import architecture_features as features
        payload = b"author-checkpoint-bytes"
        class Response:
            status_code = 206
            def raise_for_status(self):
                pass
        class Session:
            def get(self, url, headers, **kwargs):
                start, end = map(int, headers["Range"].removeprefix("bytes=").split("-"))
                result = Response()
                result.headers = {"Content-Range": f"bytes {start}-{end}/{len(payload)}"}
                result.content = payload[start:end+1]
                return result
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root/"weights.pt"
            path.with_suffix(".pt.part").write_bytes(payload[:7])
            with patch.object(features, "ROOT", root), patch.object(features.requests, "Session", Session):
                features.download("https://author.example/weights", path, size=len(payload), sha256=hashlib.sha256(payload).hexdigest())
                self.assertEqual(path.read_bytes(), payload)
                path.write_bytes(payload+b"tampered")
                with self.assertRaises(ValueError):
                    features.download("https://author.example/weights", path, sha256=hashlib.sha256(payload).hexdigest())

    def test_architecture_selection_uses_measured_oof_and_excludes_random_controls(self):
        from src.architecture_workflow import choose_candidate
        stats = {"baseline": {"mse": 100., "measured": {"mse": 100.}},
                 "a": {"mse": 50., "measured": {"mse": 20.}},
                 "b": {"mse": 200., "measured": {"mse": 10.}},
                 "random": {"mse": 1., "measured": {"mse": 1.}}}
        protocols = {p: stats for p in ("molecule", "scaffold")}
        self.assertEqual(choose_candidate(protocols, {"a": {}, "b": {}, "random": {"control_only": True}}), "b")

    def test_exact_score_needs_true_targets_and_public_mask_and_aligns_ids(self):
        prediction = pd.DataFrame({"index": [2, 1, 3], "SI VACV": [10., 4., 999.]})
        reference = pd.DataFrame({"index": [1, 2, 3], "target": [2., 12., np.nan], "is_public": [True, True, False]})
        self.assertEqual(official_reference_mse(prediction, reference), 4.)
        with self.assertRaises(ValueError):
            official_reference_mse(prediction, reference.drop(columns="is_public"))
        with self.assertRaises(ValueError):
            official_reference_mse(prediction.iloc[:2], reference)
        with self.assertRaises(ValueError):
            official_reference_mse(prediction, reference.assign(is_public="unknown"))


class Heads(unittest.TestCase):
    def setUp(self):
        self.x = np.random.default_rng(2).normal(size=(12, 8)).astype(np.float32)
        self.targets = np.column_stack([np.arange(1, 13), np.arange(1, 13)*10.]).astype(float)
        self.targets[0, 0], self.targets[1, 1] = np.nan, np.nan
        self.weights = np.linspace(.25, 1, 12)
        self.source = np.arange(12) % 2 == 0

    def test_missing_targets_do_not_enter_scaler_or_ridge_fit(self):
        normalized, mask, mean, _ = target_scaler(self.targets)
        self.assertFalse(mask[0, 0])
        self.assertAlmostEqual(mean[0], np.log(np.arange(2, 13)).mean())
        self.assertTrue(np.isfinite(normalized).all())
        model = FrozenHeads("ridge").fit(self.x, self.targets, self.weights, self.source)
        self.assertEqual(model.n_train, [11, 11])
        self.assertTrue(np.isfinite(model.predict_si(self.x)).all())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"heads.joblib"
            joblib.dump(model, path)
            np.testing.assert_array_equal(joblib.load(path).predict_si(self.x), model.predict_si(self.x))

    @unittest.skipUnless(importlib.util.find_spec("torch"), "isolated architecture environment required")
    def test_shared_and_domain_heads_have_masked_loss_and_only_fit_target_domain_when_requested(self):
        settings = {"epochs": 3, "batch_size": 4, "learning_rate": .001, "weight_decay": .01, "hidden": 8, "dropout": .1}
        for kind in ("mlp", "domain", "kaggle_only"):
            model = FrozenHeads(kind, settings=settings).fit(self.x, self.targets, self.weights, self.source)
            self.assertEqual(model.n_train, [5, 6] if kind == "kaggle_only" else [11, 11])
            prediction = model.predict_si(self.x)
            self.assertTrue(np.isfinite(prediction).all())
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory)/"model.joblib"
                joblib.dump(model, path)
                np.testing.assert_array_equal(prediction, joblib.load(path).predict_si(self.x))


class GP(unittest.TestCase):
    def test_tanimoto_does_not_overflow_uint8_and_gp_handles_partial_labels(self):
        x = np.zeros((6, 600), dtype=np.uint8)
        x[:2] = 1
        x[2, :300] = 1
        x[3, 300:] = 1
        x[4, ::2] = 1
        x[5, 1::2] = 1
        kernel = tanimoto(x, x)
        self.assertEqual(kernel[0, 1], 1.)
        self.assertEqual(kernel[2, 3], 0.)
        y = np.column_stack([np.arange(1, 7), np.arange(1, 7)*10.]).astype(float)
        y[0, 0], y[1, 1] = np.nan, np.nan
        model = JointGP(maxiter=5).fit(x, y, np.ones(6))
        self.assertEqual(model.n_train, [5, 5])
        parts = model.predict_components(x)
        self.assertTrue(np.isfinite(parts["mean"]).all())
        self.assertTrue((parts["mean"] >= parts["median"]).all())
        self.assertTrue((parts["log_si_variance"] >= 0).all())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"gp.joblib"
            joblib.dump(model, path)
            np.testing.assert_allclose(parts["mean"], joblib.load(path).predict_components(x)["mean"])
