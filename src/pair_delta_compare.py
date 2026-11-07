"""OOF evaluation of paired concentration deltas; folds are created before pairs."""

import argparse
import json
import joblib

import numpy as np
import pandas as pd

from src import splits, xgb_workflow as workflow
from src.local_compare import fold_indices
from src.local_score import score
from src.models.xgb_pair_delta import XGBPairDelta, assert_reference_partition
from src.models.xgb_two_head import exact_target
from src.data.quality_weights import row_weights


def benchmark(frame, reference, quality, matrix, config):
    options = config["pair_delta"]
    smiles = frame.standard_smiles.tolist()
    if frame.standard_smiles.duplicated().any() or reference.original_row.duplicated().any():
        raise ValueError("development structures and original-row identifiers must be unique")
    if len(matrix) != len(frame):
        raise ValueError("feature rows do not align with development structures")
    reference = reference[reference.standard_smiles.isin(smiles)].copy()
    if reference.empty:
        raise ValueError("no original competition targets in development rows")
    truth = reference.SI.to_numpy(float)
    if not np.isfinite(truth).all() or (truth < 0).any():
        raise ValueError("invalid original competition targets")
    weights = row_weights(frame, quality)
    targets = {name: exact_target(frame, name) for name in ("IC50", "CC50")}
    outputs, manifests = [], []
    for protocol in ("molecule", "scaffold"):
        for repeat in range(options["repeats"]):
            folds = fold_indices(smiles, protocol, options["folds"], config["seed"] + repeat)
            manifests.append({"protocol": protocol, "repeat": repeat, "indices": folds})
            for fold, indices in folds.items():
                held = np.asarray(indices, dtype=int)
                fit = np.setdiff1d(np.arange(len(frame)), held)
                # No held molecule or scaffold may enter the pair/reference bank.
                held_smiles = set(frame.iloc[held].standard_smiles)
                assert_reference_partition(frame.iloc[fit].standard_smiles, held_smiles,
                                           splits.murcko_scaffold, protocol)
                rows = reference[reference.standard_smiles.isin(held_smiles)].copy()
                if rows.empty:
                    raise ValueError("fold has no original competition rows")
                print(f"[pair-delta] {protocol} repeat={repeat+1} fold={int(fold)+1}; fit={len(fit)} score={len(rows)}", flush=True)
                model = XGBPairDelta(config["model"].get("params"), options.get("pair_params"),
                    options["max_references"], options["min_similarity"])
                model.fit(np.asarray(matrix)[fit], targets["IC50"][fit], targets["CC50"][fit], weights[fit])
                baseline_prediction = model.baseline.predict_si(np.asarray(matrix)[held])
                prediction = model.predict_si(np.asarray(matrix)[held])
                positions = rows.standard_smiles.map(pd.Series(np.arange(len(held)), index=frame.iloc[held].standard_smiles).to_dict()).to_numpy(int)
                rows = rows.rename(columns={"SI": "SI_true"})
                rows["pair_delta"] = prediction[positions]
                rows["weighted_baseline"] = baseline_prediction[positions]
                rows["pair_coverage"] = model.last_pair_coverage[positions]
                rows["protocol"], rows["repeat"], rows["fold"] = protocol, repeat, fold
                outputs.append(rows)
    oof = pd.concat(outputs, ignore_index=True)
    report = {"method": "median(reference log concentration + learned query-reference delta)",
        "fallback": "weighted XGB when an endpoint has no eligible reference", "folds": options["folds"],
        "repeats": options["repeats"], "holdout_used": False, "reference_bank": "fold train rows only",
        "protocols": {}}
    for protocol, rows in oof.groupby("protocol", sort=False):
        report["protocols"][protocol] = {"pair_delta": score(rows.SI_true, rows.pair_delta),
            "weighted_baseline": score(rows.SI_true, rows.weighted_baseline),
            "pair_coverage": float(rows.pair_coverage.mean()),
            "repeat_mse": [score(g.SI_true, g.pair_delta)["mse"] for _, g in rows.groupby("repeat")]}
    return report, oof, manifests


def run(config):
    parts = workflow.load_parts()
    frame = pd.concat([parts["train"], parts["validation"]], ignore_index=True)
    base = workflow.ROOT / "data/processed"
    report, oof, manifests = benchmark(frame, pd.read_csv(base / "kaggle_reference.csv"),
        pd.read_csv(base / "public_quality.csv"), workflow.features(frame), config)
    out = workflow.ROOT / "reports"
    out.mkdir(exist_ok=True)
    (out / "pair_delta_comparison.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    (out / "pair_delta_splits.json").write_text(json.dumps(manifests, indent=2), encoding="utf-8")
    oof.to_csv(out / "pair_delta_oof.csv", index=False)

    # Only after the fold evaluation, refit on the allowed development pool.
    weights = row_weights(frame, pd.read_csv(base / "public_quality.csv"))
    targets = {name: exact_target(frame, name) for name in ("IC50", "CC50")}
    final_model = XGBPairDelta(config["model"].get("params"), config["pair_delta"].get("pair_params"),
        config["pair_delta"]["max_references"], config["pair_delta"]["min_similarity"])
    matrix = workflow.features(frame)
    final_model.fit(matrix, targets["IC50"], targets["CC50"], weights)
    artifacts = workflow.ROOT / "artifacts"
    artifacts.mkdir(exist_ok=True)
    model_path = artifacts / "xgb_pair_delta.joblib"
    joblib.dump(final_model, model_path)
    training = {"config": config, "development_structures": len(frame),
        "training_rows": {name: int(np.isfinite(values).sum()) for name, values in targets.items()},
        "pair_rows": final_model.n_pairs, "holdout_used": False, "test_endpoints_used": False,
        "oof_report": "reports/pair_delta_comparison.json", "artifact": "artifacts/xgb_pair_delta.joblib"}
    (out / "xgb_pair_delta_training.json").write_text(json.dumps(training, indent=2, allow_nan=False), encoding="utf-8")
    report["artifact"] = training["artifact"]
    print(json.dumps(report, indent=2), flush=True)
    return report


def main(argv=None):
    from src.pipeline import load_config
    parser = argparse.ArgumentParser(description="Leakage-controlled paired delta OOF experiment")
    parser.add_argument("--config", default="configs/xgb_pair_delta.json")
    args = parser.parse_args(argv)
    run(load_config(args.config))


if __name__ == "__main__":
    main()
