"""Standalone Tanimoto KRR experiment; reads only the committed train/validation splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.data.standardize import standardize_smiles
from src.features.fingerprints import morgan_matrix
from src.models.tanimoto_krr import TanimotoKRR

ROOT = Path(__file__).resolve().parents[1]


def exact_target(frame: pd.DataFrame, name: str) -> np.ndarray:
    values = pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float)
    qualifier = name + "_qualifier"
    if qualifier not in frame:
        raise ValueError(f"missing {qualifier}; cannot distinguish exact from censored {name}")
    censored = frame[qualifier].astype("string").fillna("").str.strip().ne("").to_numpy()
    values[censored | ~np.isfinite(values) | (values <= 0)] = np.nan
    return values


def _read_part(path: Path) -> tuple[pd.DataFrame, np.ndarray]:
    frame = pd.read_csv(path)
    column = "standard_smiles" if "standard_smiles" in frame else "SMILES"
    if column not in frame:
        raise ValueError(f"{path}: missing standard_smiles or SMILES")
    smiles = [standardize_smiles(value) for value in frame[column]]
    if any(value is None for value in smiles):
        raise ValueError(f"{path}: invalid SMILES")
    return frame, morgan_matrix(smiles, radius=2, bits=2048)


def run(config_path: str | Path) -> dict:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    if config.get("features") != {"kind": "morgan", "radius": 2, "bits": 2048}:
        raise ValueError("Tanimoto KRR requires binary Morgan radius 2 fingerprints with 2048 bits")
    if config.get("model", {}).get("strategy") != "tanimoto_krr":
        raise ValueError("model.strategy must be tanimoto_krr")
    if config.get("data") != {
        "train": "data/processed/train.csv",
        "validation": "data/processed/validation.csv",
    }:
        raise ValueError("this experiment is restricted to the committed train/validation splits")
    train_path = ROOT / config["data"]["train"]
    validation_path = ROOT / config["data"]["validation"]
    train, train_fp = _read_part(train_path)
    validation, validation_fp = _read_part(validation_path)
    model = TanimotoKRR(config["model"].get("params"), config["model"].get("clip_si"))
    model.fit(train_fp, exact_target(train, "IC50"), exact_target(train, "CC50"))
    predictions = model.predict_si(validation_fp)

    si = pd.to_numeric(validation["SI"], errors="coerce").to_numpy(float)
    source = validation.get("source", pd.Series("", index=validation.index)).fillna("").astype(str)
    competition = source.str.split(";").apply(lambda values: "kaggle_train" in values).to_numpy()
    mask = competition & np.isfinite(si) & (si > 0)
    report = {
        "experiment": config["name"],
        "training_rows": len(train),
        "validation_rows": len(validation),
        "validation_competition_targets": int(mask.sum()),
        "validation_raw_si_mse": float(np.mean((si[mask] - predictions[mask]) ** 2)) if mask.any() else None,
        "n_train": model.n_train,
        "holdout_used": False,
        "test_data_used": False,
        "model": config["model"],
        "feature": config["features"],
    }
    model_path, report_path = (ROOT / config["outputs"][name] for name in ("model", "report"))
    model_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "config": config}, model_path)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=ROOT / "configs/tanimoto_krr.json")
    args = parser.parse_args(argv)
    run(args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
