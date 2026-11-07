"""User-authorized LOCAL endpoint diagnostic AFTER inference; never imported by model code.

The derived ratio is not the hidden official target. Missing/censored endpoints
remain unscored; no organizer label conventions are guessed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.data.schema import parse_measurement
from src.local_score import score


def reference_from_endpoints(test, id_column="index"):
    required = [id_column, "CC50_uM", "IC50_uM_VACV"]
    if any(name not in test for name in required):
        raise ValueError(f"Diagnostic requires {required}")
    identifiers = test[id_column].astype("string")
    if identifiers.isna().any() or identifiers.str.strip().eq("").any() or identifiers.duplicated().any():
        raise ValueError("Test IDs must be unique and nonempty")
    targets, reasons = [], []
    for cc_raw, ic_raw in zip(test.CC50_uM, test.IC50_uM_VACV):
        cc, cc_mark = parse_measurement(cc_raw)
        ic, ic_mark = parse_measurement(ic_raw)
        reason = "scorable"
        target = np.nan
        if cc_mark or ic_mark:
            reason = "censored_or_approximate_endpoint"
        elif cc is None and ic is None:
            reason = "missing_both"
        elif cc is None:
            reason = "missing_CC50"
        elif ic is None:
            reason = "missing_IC50"
        elif not np.isfinite([cc, ic]).all():
            reason = "nonfinite_endpoint"
        elif cc <= 0 or ic <= 0:
            reason = "nonpositive_endpoint"
        else:
            target = cc / ic
            if not np.isfinite(target):
                reason, target = "nonfinite_ratio", np.nan
        targets.append(target)
        reasons.append(reason)
    return pd.DataFrame({id_column: identifiers, "ratio_target": targets, "status": reasons})


def evaluate_predictions(reference, prediction, id_column="index", prediction_column="SI VACV"):
    if id_column not in prediction or prediction_column not in prediction:
        raise ValueError("Missing submission ID or prediction column")
    prediction = prediction[[id_column, prediction_column]].copy()
    prediction[id_column] = prediction[id_column].astype("string")
    if prediction[id_column].isna().any() or prediction[id_column].duplicated().any():
        raise ValueError("Submission IDs must be unique and nonempty")
    if set(prediction[id_column]) != set(reference[id_column]):
        raise ValueError("Submission IDs must exactly match ALL test IDs, including unscorable rows")
    prediction[prediction_column] = pd.to_numeric(prediction[prediction_column], errors="coerce")
    values = prediction[prediction_column].to_numpy(float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("All submission predictions must be finite and nonnegative")
    merged = reference.merge(prediction, on=id_column, how="left", validate="one_to_one", sort=False)
    valid = merged.status.eq("scorable")
    if not valid.any():
        raise ValueError("No exact positive endpoint pairs to evaluate")
    metrics = score(merged.loc[valid, "ratio_target"], merged.loc[valid, prediction_column])
    return {"n_test": len(merged), "n_evaluated": int(valid.sum()),
        "coverage": float(valid.mean()), "mse_ratio_subset": metrics["mse"],
        "rmse_ratio_subset": metrics["rmse"], "mae_ratio_subset": metrics["mae"],
        "full_test_ratio_mse": metrics["mse"] if valid.all() else None,
        "official_kaggle_score": None}


def run(test_path, submissions, output_prefix, id_column="index", prediction_column="SI VACV"):
    test_path, output_prefix = Path(test_path), Path(output_prefix)
    # This module only reads endpoints for diagnostics. Inference still reads IDs + SMILES only.
    test = pd.read_csv(test_path, usecols=[id_column, "CC50_uM", "IC50_uM_VACV"], dtype={id_column: "string"})
    reference = reference_from_endpoints(test, id_column)
    rows = []
    for filename in submissions:
        path = Path(filename)
        predicted = pd.read_csv(path, usecols=[id_column, prediction_column], dtype={id_column: "string"})
        values = evaluate_predictions(reference, predicted, id_column, prediction_column)
        values.update({"submission": str(path), "submission_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        rows.append(values)
    if not rows:
        raise ValueError("No submission files supplied")
    report = {"purpose": "LOCAL diagnostic only; original official test targets unavailable",
        "formula": "mean((SI_pred - CC50_uM/IC50_uM_VACV)**2) on exact positive endpoint pairs",
        "limitations": ["unscorable rows excluded identically for every submission",
            "official target conventions unknown", "public leaderboard test subset unknown",
            "not an official Kaggle score; not used by fitting, feature extraction, or automatic promotion"],
        "test_sha256": hashlib.sha256(test_path.read_bytes()).hexdigest(),
        "coverage_by_status": {str(key): int(value) for key, value in reference.status.value_counts().items()},
        "results": rows}
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    # Aggregate reports only; derived test targets are never exported.
    output_prefix.with_suffix(".json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    table = pd.DataFrame(rows).sort_values("mse_ratio_subset")
    table.to_csv(output_prefix.with_suffix(".csv"), index=False)
    print("[LOCAL TEST DIAGNOSTIC] " + json.dumps(report["coverage_by_status"]), flush=True)
    print(table[["submission", "n_evaluated", "n_test", "mse_ratio_subset"]].to_string(index=False), flush=True)
    print("[LOCAL TEST DIAGNOSTIC] Partial ratio-reference MSE; NOT official Kaggle score.", flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description="Local CC50/IC50 error diagnostic after inference; no training or submission")
    parser.add_argument("--test", default="data/raw/kaggle/test.csv")
    parser.add_argument("--submission", nargs="+", help="Existing predictions; defaults to all reports/submission*.csv except member diagnostics")
    parser.add_argument("--output-prefix", default="reports/test_diagnostics")
    parser.add_argument("--id-column", default="index")
    parser.add_argument("--prediction-column", default="SI VACV")
    args = parser.parse_args(argv)
    paths = args.submission or [str(p) for p in sorted(Path("reports").glob("submission*.csv")) if not p.stem.endswith("_members")]
    run(args.test, paths, args.output_prefix, args.id_column, args.prediction_column)


if __name__ == "__main__":
    main()
