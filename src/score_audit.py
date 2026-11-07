"""Audit score reproducibility; never fit a surrogate to leaderboard observations."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.local_score import score
from src.test_diagnostics import run as diagnose

ROOT = Path(__file__).resolve().parents[1]


def official_reference_mse(prediction, reference):
    """Exact public MSE ONLY if organizer ground truth AND public mask are supplied."""
    required = {"index", "target", "is_public"}
    if not required.issubset(reference.columns):
        raise ValueError("Exact scoring requires organizer index, target and is_public columns")
    ref = reference.copy()
    pred = prediction.copy()
    for frame in (ref, pred):
        frame["index"] = frame["index"].astype("string")
        if frame["index"].isna().any() or frame["index"].duplicated().any():
            raise ValueError("Unique nonempty IDs required")
    if set(ref["index"]) != set(pred["index"]):
        raise ValueError("Prediction/reference ID sets differ")
    predictions = pd.to_numeric(pred["SI VACV"], errors="coerce").to_numpy(float)
    if not np.isfinite(predictions).all() or (predictions < 0).any():
        raise ValueError("Every prediction must be finite and nonnegative")
    flags = ref.is_public.astype("string").str.lower()
    if not flags.isin(["true", "false"]).all():
        raise ValueError("Explicit Boolean public mask required")
    merged = ref.merge(pred[["index", "SI VACV"]], on="index", validate="one_to_one")
    public = merged.is_public.astype("string").str.lower().eq("true")
    if not public.any():
        raise ValueError("Empty public subset")
    return score(merged.loc[public, "target"], merged.loc[public, "SI VACV"])["mse"]


def run():
    official = json.loads((ROOT / "reports/hypotheses/kaggle_submissions.json").read_text(encoding="utf-8"))["submissions"]
    known = [{"path": "reports/submission_xgb_weighted.csv", "ref": 58079620,
        "sha256": "92a0ea69f93cc5c48663ec0b84d68f853ae4c486c5d0b55ee713f5c85b787169", "public_score": 1967765.75828}]
    known += [{"path": r["submission"], "ref": r["ref"], "sha256": r["submission_sha256"],
               "public_score": r["public_score"]} for r in official]
    for row in known:
        if hashlib.sha256((ROOT / row["path"]).read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError("Historical official score cannot be attached to changed predictions")
    diagnostics = diagnose(ROOT / "data/raw/kaggle/test.csv", [ROOT / r["path"] for r in known],
                           ROOT / "reports/score_audit_test_diagnostics")
    rows = []
    for record, values in zip(known, diagnostics["results"]):
        rows.append({**record, "local_subset_mse": values["mse_ratio_subset"],
            "official_minus_local": record["public_score"]-values["mse_ratio_subset"],
            "coverage": f"{values['n_evaluated']}/{values['n_test']}",
            "official_source": "Previously scored exact submission bytes; lookup, not local recomputation"})
    result = {"can_recompute_exact_official_for_new_predictions": False,
        "missing_inputs": ["organizer test ground truth for all official-scored rows", "organizer public/private row mask"],
        "competition_files_available": ["train_v2.csv", "test.csv", "sample_submit.csv"],
        "metric_formula": "mean((prediction-target)**2)", "local_reference": "468 exact endpoint ratios out of 597 IDs",
        "identity_requirement": "Same true targets, same evaluated IDs, same aggregation; formula alone is insufficient",
        "no_leaderboard_calibration": True, "known_submissions": rows,
        "selection_for_architectures": "Original measured TRAIN SI OOF MSE, molecule and scaffold protocols; conditional labels reported separately"}
    directory = ROOT / "reports/architectures"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "score_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(directory / "score_audit.csv", index=False)
    print(pd.DataFrame(rows)[["ref", "local_subset_mse", "public_score", "coverage"]].to_string(index=False))
    print("Exact recomputation unavailable: organizer targets and public mask are missing. No score correction fitted.")


if __name__ == "__main__":
    run()
