"""Recover unchanged concentration heads from the portable hybrid checkpoint; no fit."""
import joblib
import numpy as np
import pandas as pd
from src import xgb_workflow as workflow
from src.models.xgb_two_head import XGBTwoHead
from src.pipeline import load_config, ROOT


def main():
    source = joblib.load(ROOT / "artifacts/xgb_hybrid.joblib")
    hybrid = source["model"]
    model = XGBTwoHead(hybrid.params, hybrid.clip_si)
    model.heads = hybrid.heads
    model.n_train = {key: value for key, value in hybrid.n_train.items() if not key.startswith("classifier")}
    model.n_features_in_ = hybrid.n_features_in_
    validation = pd.read_csv(ROOT / "data/processed/validation.csv")
    matrix = workflow.features(validation)
    original = pd.read_csv(ROOT / "reports/xgb_validation.csv")
    if not original.standard_smiles.equals(validation.standard_smiles):
        raise ValueError("validation structure order differs from original report")
    np.testing.assert_allclose(model.predict_si(matrix), original.SI_pred, rtol=1e-7, atol=1e-8)
    config = load_config(ROOT / "configs/baseline.json")
    bundle = {"model": model, "config": config, "data_sha256": workflow.data_hashes(),
        "constant_si": source["constant_si"], "versions": source["versions"],
        "recovered_from": "artifacts/xgb_hybrid.joblib; identical to original XGBoost validation predictions"}
    joblib.dump(bundle, ROOT / config["submission"]["model_file"])
    workflow.submission(config)
    print("Baseline recovered without training; original validation predictions reproduced.")


if __name__ == "__main__":
    main()
