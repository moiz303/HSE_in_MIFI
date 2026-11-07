"""Development-only OOF comparison of predeclared bounded tail weight profiles."""
import json
import time
import math
import numpy as np
import pandas as pd
from src import xgb_workflow as workflow, splits
from src.local_compare import fold_indices
from src.local_score import score, paired_interval
from src.models.xgb_two_head import exact_target
from src.models.xgb_tail import XGBTail, PROFILES, tail_weights
from src.data.quality_weights import row_weights


def choose_profile(protocols):
    eligible = []
    for candidate in list(PROFILES)[1:]:
        baseline, proposed = protocols["molecule"]["weighted"], protocols["molecule"][candidate]
        if baseline["mse"] <= 0 or proposed["mse"] > .98 * baseline["mse"]:
            continue
        if sum(n < b for n, b in zip(proposed["repeat_mse"], baseline["repeat_mse"])) < math.ceil(2*len(baseline["repeat_mse"])/3):
            continue
        safe = True
        for protocol, variants in protocols.items():
            base, new = variants["weighted"], variants[candidate]
            if protocol == "scaffold" and new["mse"] > 1.02 * base["mse"]:
                safe = False
            for subset in ("measured", "low_si"):
                if base[subset] is not None and new[subset]["mse"] > 1.02 * base[subset]["mse"]:
                    safe = False
        if safe:
            eligible.append(candidate)
    selected = min(eligible, key=lambda n: protocols["molecule"][n]["mse"]) if eligible else "weighted"
    return {"selected": selected,
        "rule": ">=2% molecule MSE gain; wins in >=2/3 repeats; <=2% scaffold, measured and SI<=100 deterioration",
        "note": "Exploratory OOF selection, not an independent estimate or official Kaggle score"}


def benchmark(frame, reference, quality, matrix, params=None, folds=3, repeats=3, seed=20261106):
    if not 2 <= folds <= 10 or not 1 <= repeats <= 20:
        raise ValueError("folds must be 2..10; repeats must be 1..20")
    if frame.standard_smiles.duplicated().any() or reference.original_row.duplicated().any():
        raise ValueError("Duplicate structure or reference key")
    reference = reference[reference.standard_smiles.isin(frame.standard_smiles)].copy()
    if reference.empty or len(matrix) != len(frame):
        raise ValueError("Missing reference rows or misaligned features")
    score(reference.SI, np.zeros(len(reference)))
    targets = {n: exact_target(frame, n) for n in ("IC50", "CC50")}
    reliability = row_weights(frame, quality)
    matrix = np.asarray(matrix)
    outputs, manifests = [], []
    for protocol in ("molecule", "scaffold"):
        for repeat in range(repeats):
            split = fold_indices(frame.standard_smiles.tolist(), protocol, folds, seed+repeat)
            manifests.append({"protocol": protocol, "repeat": repeat, "seed": seed+repeat, "indices": split})
            for fold, indices in split.items():
                test = np.asarray(indices, dtype=int)
                train = np.setdiff1d(np.arange(len(frame)), test)
                rows = reference[reference.standard_smiles.isin(frame.iloc[test].standard_smiles)].copy()
                if rows.empty:
                    raise ValueError("No original targets in fold")
                positions = rows.standard_smiles.map(pd.Series(np.arange(len(test)), index=frame.iloc[test].standard_smiles)).to_numpy(int)
                rows = rows.rename(columns={"SI": "SI_true"})
                print(f"[tail] {protocol} repeat={repeat+1}/{repeats} fold={int(fold)+1}/{folds}; train={len(train)}, score={len(rows)}", flush=True)
                for profile in PROFILES:
                    # Target-dependent weights see only the training portion of this fold.
                    weights = tail_weights(frame.iloc[train], reliability[train], profile)
                    model = XGBTail(params).fit(matrix[train], targets["IC50"][train], targets["CC50"][train], sample_weight=weights)
                    rows[profile] = model.predict_si(matrix[test])[positions]
                    print(f"[tail] {profile}: MSE={score(rows.SI_true, rows[profile])['mse']:.6f}; boosted IC50 rows={int((weights['IC50'] > reliability[train]).sum())}", flush=True)
                rows["protocol"], rows["repeat"], rows["fold"] = protocol, repeat, fold
                rows["scaffold"] = rows.standard_smiles.map(splits.murcko_scaffold)
                outputs.append(rows)
    oof = pd.concat(outputs, ignore_index=True)
    report = {"metric": "raw SI MSE on original competition OOF rows", "folds": folds, "repeats": repeats,
        "seed": seed, "holdout_used": False, "development_structures": len(frame),
        "original_competition_rows": len(reference), "profiles": PROFILES, "protocols": {}}
    for protocol, rows in oof.groupby("protocol", sort=False):
        if len(rows) != len(reference)*repeats or not rows.groupby("original_row").size().eq(repeats).all():
            raise AssertionError("Each original row must be predicted exactly once per repeat")
        subsets = {"measured": rows.target_is_label.astype("string").str.lower().eq("false").fillna(False),
                   "low_si": rows.SI_true.le(100), "high_si": rows.SI_true.gt(100)}
        variants = {}
        for candidate in PROFILES:
            values = score(rows.SI_true, rows[candidate])
            values["repeat_mse"] = [score(p.SI_true, p[candidate])["mse"] for _, p in rows.groupby("repeat")]
            for name, mask in subsets.items():
                values[name] = score(rows.loc[mask, "SI_true"], rows.loc[mask, candidate]) if mask.any() else None
            if candidate != "weighted":
                values["paired_gain_vs_weighted"] = paired_interval(rows, candidate, baseline="weighted")
            variants[candidate] = values
        report["protocols"][protocol] = variants
    report["selection"] = choose_profile(report["protocols"])
    return report, oof, manifests


def run(config, frame, matrix, quality):
    started = time.monotonic()
    reference = pd.read_csv(workflow.ROOT / "data/processed/kaggle_reference.csv")
    settings = config.get("tail_comparison", {})
    print("[tail] Four profiles; fresh fit per fold/profile; test endpoints and holdout excluded", flush=True)
    report, oof, manifests = benchmark(frame, reference, quality, matrix, config["model"].get("params"),
        settings.get("folds", 3), settings.get("repeats", 3), config.get("seed", 20261106))
    import xgboost
    from rdkit import rdBase
    report.update({"config": config, "data_sha256": workflow.data_hashes(config),
                   "versions": {"xgboost": xgboost.__version__, "rdkit": rdBase.rdkitVersion},
                   "elapsed_seconds": time.monotonic()-started})
    directory = workflow.ROOT / "reports"
    directory.mkdir(exist_ok=True)
    (directory / "tail_comparison.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    (directory / "tail_comparison_splits.json").write_text(json.dumps({"development_smiles": frame.standard_smiles.tolist(), "splits": manifests}, indent=2), encoding="utf-8")
    oof.to_csv(directory / "tail_comparison_oof.csv", index=False)
    table = pd.DataFrame([{"protocol": p, "candidate": c, "mse": v["mse"],
        "measured_mse": v["measured"]["mse"] if v["measured"] else None,
        "low_si_mse": v["low_si"]["mse"] if v["low_si"] else None}
        for p, variants in report["protocols"].items() for c, v in variants.items()])
    table.to_csv(directory / "tail_comparison.csv", index=False)
    print(table.to_string(index=False), flush=True)
    print("[tail selection] " + json.dumps(report["selection"]), flush=True)
    return report
