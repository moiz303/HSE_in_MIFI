"""Real CatBoost fits on tiny synthetic data only."""
import numpy as np
from catboost import CatBoostRegressor
from src import xgb_workflow as workflow
from src.models.catboost_two_head import CatBoostTwoHead
from tests.test_xgb_two_head import ToyWorkflow as _ToyWorkflow


class CatBoostWorkflow(_ToyWorkflow):
    def setUp(self):
        super().setUp()
        self.config["model"] = {"strategy": "catboost_two_head", "clip_si": None,
            "params": {"iterations": 4, "depth": 2, "thread_count": 1}}
        self.config["submission"]["model_file"] = "artifacts/catboost_two_head.joblib"

    def test_native_heads_and_independent_masks(self):
        x = np.arange(48, dtype=float).reshape(12, 4)
        ic = np.arange(1, 13, dtype=float)
        cc = ic * 10
        ic[:2] = np.nan
        cc[-3:] = np.nan
        model = CatBoostTwoHead(self.config["model"]["params"]).fit(x, ic, cc)
        self.assertEqual(model.n_train["IC50"], 10)
        self.assertEqual(model.n_train["CC50"], 9)
        logs = model.predict_log_concentrations(x)
        np.testing.assert_allclose(model.predict_si(x), 10 ** (logs["CC50"] - logs["IC50"]))
        with self.assertRaises(ValueError):
            CatBoostTwoHead({"n_estimators": 5})
        workflow.train(self.config)
        bundle = workflow.load_bundle(self.config)
        for name, head in bundle["model"].heads.items():
            restored = CatBoostRegressor()
            restored.load_model(str(self.root / f"artifacts/catboost_two_head_{name}.cbm"))
            matrix = workflow.features(self.frame)
            np.testing.assert_allclose(head.predict(matrix), restored.predict(matrix))


del _ToyWorkflow
