"""Train-only OOF choice of a bounded analogue correction to the weighted baseline."""
import json
import math
import time
import numpy as np
import pandas as pd
from src import xgb_workflow as workflow, splits
from src.local_compare import fold_indices
from src.local_score import score, paired_interval
from src.models.xgb_two_head import exact_target
from src.models.xgb_analog import XGBAnalog, NEIGHBOR_SETTINGS
from src.data.quality_weights import row_weights

ALPHAS = {"weighted": 0.0, "alpha_010": .1, "alpha_025": .25, "alpha_050": .5}


def choose_alpha(protocols):
    """Main-protocol gain with scaffold and measured-target safeguards; no test targets."""
    eligible = []
    for candidate in list(ALPHAS)[1:]:
        molecule = protocols["molecule"]
        base, values = molecule["weighted"], molecule[candidate]
        if base["mse"] <= 0 or values["mse"] > .98 * base["mse"]:
            continue
        wins = sum(new < old for new, old in zip(values["repeat_mse"], base["repeat_mse"]))
        if wins < math.ceil(2*len(base["repeat_mse"])/3):
            continue
        safe = True
        for protocol, variants in protocols.items():
            baseline, proposed = variants["weighted"], variants[candidate]
            if protocol == "scaffold" and proposed["mse"] > 1.02 * baseline["mse"]:
                safe = False
            measured = baseline.get("measured")
            if measured is not None and proposed["measured"]["mse"] > 1.02 * measured["mse"]:
                safe = False
        if safe:
            eligible.append(candidate)
    winner = min(eligible, key=lambda name: (protocols["molecule"][name]["mse"], ALPHAS[name])) if eligible else "weighted"
    return {"alpha": ALPHAS[winner], "selected": winner,
        "rule": ">=2% molecule OOF MSE gain; wins in >=2/3 repeats; <=2% scaffold and measured-target deterioration",
        "note": "Exploratory training-OOF selection; not an independent performance claim or Kaggle guarantee"}


def benchmark(frame, reference, quality, matrix, params=None, folds=3, repeats=3, seed=20261106):
    if not 2 <= folds <= 10 or not 1 <= repeats <= 20:
        raise ValueError("folds must be 2..10; repeats must be 1..20")
    if frame.standard_smiles.duplicated().any() or reference.original_row.duplicated().any():
        raise ValueError("Duplicate structure or original-row keys")
    reference = reference[reference.standard_smiles.isin(frame.standard_smiles)].copy()
    if reference.empty:
        raise ValueError("No original competition rows")
    score(reference.SI, np.zeros(len(reference)))
    matrix = np.asarray(matrix)
    if len(matrix) != len(frame):
        raise ValueError("Features are not aligned")
    targets = {name: exact_target(frame, name) for name in ("IC50", "CC50")}
    weights = row_weights(frame, quality)
    output, manifests = [], []
    for protocol in ("molecule", "scaffold"):
        for repeat in range(repeats):
            split = fold_indices(frame.standard_smiles.tolist(), protocol, folds, seed+repeat)
            manifests.append({"protocol": protocol, "repeat": repeat, "seed": seed+repeat, "indices": split})
            for name, indices in split.items():
                test = np.asarray(indices, dtype=int)
                train = np.setdiff1d(np.arange(len(frame)), test)
                rows = reference[reference.standard_smiles.isin(frame.iloc[test].standard_smiles)].copy()
                if rows.empty:
                    raise ValueError("No original competition targets in fold")
                print(f"[analogs] {protocol} repeat={repeat+1}/{repeats} fold={int(name)+1}/{folds}; fit={len(train)}, score={len(rows)}", flush=True)
                model = XGBAnalog(params).fit(matrix[train], targets["IC50"][train], targets["CC50"][train],
                                             sample_weight=weights[train])
                components = model.predict_components(matrix[test])
                positions = rows.standard_smiles.map(pd.Series(np.arange(len(test)), index=frame.iloc[test].standard_smiles)).to_numpy(int)
                rows = rows.rename(columns={"SI": "SI_true"})
                for candidate, alpha in ALPHAS.items():
                    predictions = components["SI_baseline"] + alpha*(components["SI_analog"]-components["SI_baseline"])
                    rows[candidate] = predictions[positions]
                rows["analog_available"] = components["analog_available"][positions]
                rows["protocol"], rows["repeat"], rows["fold"] = protocol, repeat, name
                rows["scaffold"] = rows.standard_smiles.map(splits.murcko_scaffold)
                print("[analogs] MSE " + json.dumps({candidate: score(rows.SI_true, rows[candidate])["mse"] for candidate in ALPHAS})
                      + f"; coverage={rows.analog_available.mean():.1%}", flush=True)
                output.append(rows)
    oof = pd.concat(output, ignore_index=True)
    report = {"metric": "raw SI MSE on original competition OOF rows", "folds": folds, "repeats": repeats,
        "seed": seed, "holdout_used": False, "development_structures": len(frame),
        "original_competition_rows": len(reference), "neighbor_settings": dict(NEIGHBOR_SETTINGS), "protocols": {}}
    for protocol, rows in oof.groupby("protocol", sort=False):
        if len(rows) != len(reference)*repeats or not rows.groupby("original_row").size().eq(repeats).all():
            raise AssertionError("Each original row must be scored once per repeat")
        variants = {}
        measured = ~rows.target_is_label.astype("string").str.lower().eq("true")
        for candidate in ALPHAS:
            values = score(rows.SI_true, rows[candidate])
            values["repeat_mse"] = [score(part.SI_true, part[candidate])["mse"] for _, part in rows.groupby("repeat")]
            values["measured"] = score(rows.loc[measured, "SI_true"], rows.loc[measured, candidate]) if measured.any() else None
            values["analog_coverage"] = float(rows.analog_available.mean())
            if candidate != "weighted":
                values["paired_gain_vs_weighted"] = paired_interval(rows, candidate, baseline="weighted")
            variants[candidate] = values
        report["protocols"][protocol] = variants
    report["selection"] = choose_alpha(report["protocols"])
    return report, oof, manifests


def run(config, frame, matrix, quality):
    started = time.monotonic()
    reference = pd.read_csv(workflow.ROOT / "data/processed/kaggle_reference.csv")
    settings = config.get("analog_comparison", {})
    print(f"[analogs] {settings.get('folds', 3)} folds x {settings.get('repeats', 3)} repeats x 2 protocols; "
          "one concentration pair per fold, four mixture weights reuse the same predictions", flush=True)
    report, oof, manifests = benchmark(frame, reference, quality, matrix, config["model"].get("params"),
        settings.get("folds", 3), settings.get("repeats", 3), config.get("seed", 20261106))
    import xgboost
    from rdkit import rdBase
    report.update({"config": config, "data_sha256": workflow.data_hashes(config),
                   "versions": {"xgboost": xgboost.__version__, "rdkit": rdBase.rdkitVersion},
                   "elapsed_seconds": time.monotonic()-started})
    directory = workflow.ROOT / "reports"
    directory.mkdir(exist_ok=True)
    (directory / "analog_comparison.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    (directory / "analog_comparison_splits.json").write_text(json.dumps({"development_smiles": frame.standard_smiles.tolist(),
        "splits": manifests}, indent=2), encoding="utf-8")
    oof.to_csv(directory / "analog_comparison_oof.csv", index=False)
    table = pd.DataFrame([{"protocol": protocol, "candidate": candidate, "alpha": ALPHAS[candidate],
        "mse": metrics["mse"], "analog_coverage": metrics["analog_coverage"]}
        for protocol, variants in report["protocols"].items() for candidate, metrics in variants.items()])
    table.to_csv(directory / "analog_comparison.csv", index=False)
    print(table.to_string(index=False), flush=True)
    print("[analogs selection] " + json.dumps(report["selection"]), flush=True)
    return report
