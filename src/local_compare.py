"""Leakage-controlled OOF comparison of all three options. Run real fits on Mac only."""
import argparse
import json
import time
from collections import defaultdict
import numpy as np
import pandas as pd
from src import splits, xgb_workflow as workflow
from src.data.quality_weights import row_weights
from src.local_score import score, paired_interval
from src.models.xgb_ensemble import XGBEnsemble
from src.models.xgb_weighted import XGBWeightedTwoHead
from src.models.xgb_two_head import exact_target


def fold_indices(smiles, protocol, folds, seed):
    if protocol not in ("molecule", "scaffold") or folds < 2:
        raise ValueError("Use molecule/scaffold protocol and >=2 folds")
    groups = defaultdict(list)
    for index, value in enumerate(smiles):
        key = value if protocol == "molecule" else splits.murcko_scaffold(value)
        if key == "invalid":
            raise ValueError("Invalid scaffold input")
        groups[key].append(index)
    if len(groups) < folds:
        raise ValueError("Insufficient independent groups for requested folds")
    items = list(groups.values())
    np.random.default_rng(seed).shuffle(items)
    items.sort(key=len, reverse=True)
    result = {str(index): [] for index in range(folds)}
    for indices in items:
        destination = min(result, key=lambda name: len(result[name]))
        result[destination].extend(indices)
    # Scaffold protocol also protects against identical structures across sources.
    if protocol == "scaffold":
        splits.assert_no_leaks(smiles, result)
    else:
        seen = {}
        for name, indices in result.items():
            for index in indices:
                if smiles[index] in seen and seen[smiles[index]] != name:
                    raise AssertionError("Duplicate molecule crossed fold boundary")
                seen[smiles[index]] = name
    return result


def benchmark(frame, reference, quality, matrix, params=None, folds=3, repeats=3, seed=20261106):
    if not 2 <= folds <= 10 or not 1 <= repeats <= 20:
        raise ValueError("folds must be 2..10; repeats must be 1..20")
    if frame.standard_smiles.duplicated().any() or reference.original_row.duplicated().any():
        raise ValueError("Require unique structure rows and unique original row identifiers")
    reference = reference[reference.standard_smiles.isin(frame.standard_smiles)].copy()
    if reference.empty:
        raise ValueError("No original competition targets in development data")
    score(reference.SI.to_numpy(), np.zeros(len(reference)))  # validate, never silently drop rows
    matrix = np.asarray(matrix)
    if len(matrix) != len(frame):
        raise ValueError("Features do not align with development rows")
    weights = row_weights(frame, quality)
    targets = {name: exact_target(frame, name) for name in ("IC50", "CC50")}
    all_rows, fold_reports, manifests = [], [], []
    for protocol in ("molecule", "scaffold"):
        for repeat in range(repeats):
            split = fold_indices(frame.standard_smiles.tolist(), protocol, folds, seed+repeat)
            manifests.append({"protocol": protocol, "repeat": repeat, "seed": seed+repeat,
                              "indices_in_development_pool": split})
            for name, indices in split.items():
                test = np.asarray(indices, dtype=int)
                train = np.setdiff1d(np.arange(len(frame)), test)
                test_frame = frame.iloc[test]
                rows = reference[reference.standard_smiles.isin(test_frame.standard_smiles)].copy()
                if rows.empty:
                    raise ValueError("Fold has no competition targets; reduce fold count")
                training_ref = reference[reference.standard_smiles.isin(frame.iloc[train].standard_smiles)]
                if training_ref.empty:
                    raise ValueError("No competition targets for train-only constant")
                print(f"[local score] {protocol} repeat={repeat+1}/{repeats} fold={int(name)+1}/{folds}; "
                      f"fit={len(train)}, validation={len(test)}, competition rows={len(rows)}", flush=True)
                ensemble = XGBEnsemble(params).fit(matrix[train], targets["IC50"][train], targets["CC50"][train])
                weighted = XGBWeightedTwoHead(params).fit(matrix[train], targets["IC50"][train],
                    targets["CC50"][train], sample_weight=weights[train])
                predictions = {"variant1": ensemble.members[0].predict_si(matrix[test]),
                    "variant2": ensemble.predict_si(matrix[test]), "variant3": weighted.predict_si(matrix[test])}
                lookup = pd.Series(np.arange(len(test)), index=test_frame.standard_smiles)
                positions = rows.standard_smiles.map(lookup).to_numpy(int)
                rows = rows.rename(columns={"SI": "SI_true"})
                for variant, prediction in predictions.items():
                    rows[variant] = prediction[positions]
                rows["constant"] = float(training_ref.SI.mean())
                rows["protocol"], rows["repeat"], rows["fold"] = protocol, repeat, name
                rows["scaffold"] = rows.standard_smiles.map(splits.murcko_scaffold)
                summary = {variant: score(rows.SI_true, rows[variant]) for variant in (*predictions, "constant")}
                print("[local score] MSE " + json.dumps({v: s["mse"] for v, s in summary.items()}), flush=True)
                fold_reports.append({"protocol": protocol, "repeat": repeat, "fold": name, "scores": summary})
                all_rows.append(rows)
    oof = pd.concat(all_rows, ignore_index=True)
    report = {"metric": "mean((SI_pred-SI_original_Kaggle)**2), raw units; lower is better",
        "note": "OOF development comparison, not a predictor of exact public leaderboard score; no automatic model promotion",
        "folds": folds, "repeats": repeats, "seed": seed, "development_structures": len(frame),
        "original_competition_rows": len(reference), "holdout_used": False,
        "fold_scores": fold_reports, "protocols": {}}
    for protocol, rows in oof.groupby("protocol", sort=False):
        summary = {}
        expected = len(reference)*repeats
        if len(rows) != expected or not rows.groupby("original_row").size().eq(repeats).all():
            raise AssertionError("Every original row must be scored exactly once per repeat")
        for variant in ("variant1", "variant2", "variant3", "constant"):
            values = score(rows.SI_true, rows[variant])
            constant_mse = score(rows.SI_true, rows.constant)["mse"]
            values["skill_vs_constant"] = 1-values["mse"]/constant_mse if constant_mse > 0 else None
            values["repeat_mse"] = [score(group.SI_true, group[variant])["mse"] for _, group in rows.groupby("repeat")]
            values["repeat_mse_std"] = float(np.std(values["repeat_mse"]))
            labels = rows.target_is_label.astype("string").str.lower().eq("true")
            values["strata"] = {}
            for label, mask in {"label": labels, "measured": ~labels, "SI_lt_1000": rows.SI_true < 1000,
                                "SI_ge_10000": rows.SI_true >= 10000}.items():
                if mask.any():
                    values["strata"][label] = score(rows.loc[mask, "SI_true"], rows.loc[mask, variant])
            if variant in ("variant2", "variant3"):
                values["paired_gain_vs_variant1"] = paired_interval(rows, variant)
            summary[variant] = values
        report["protocols"][protocol] = summary
    return report, oof, manifests


def run(config):
    started = time.monotonic()
    parts = workflow.load_parts()
    frame = pd.concat([parts["train"], parts["validation"]], ignore_index=True)
    directory = workflow.ROOT / "data/processed"
    reference = pd.read_csv(directory / "kaggle_reference.csv")
    quality = pd.read_csv(directory / "public_quality.csv")
    settings = config.get("local_comparison", {})
    folds, repeats = settings.get("folds", 3), settings.get("repeats", 3)
    print(f"[local score] workload: 2 protocols x {repeats} repeats x {folds} folds; "
          "each fold fits 5 ordinary pairs + 1 weighted pair. Variant 1 reuses the first ensemble member.", flush=True)
    report, oof, manifest = benchmark(frame, reference, quality, workflow.features(frame),
        params=config["model"].get("params"), folds=folds,
        repeats=repeats, seed=config.get("seed", 20261106))
    import xgboost
    from rdkit import rdBase
    report.update({"data_sha256": workflow.data_hashes(config), "config": config,
        "versions": {"xgboost": xgboost.__version__, "rdkit": rdBase.rdkitVersion},
        "elapsed_seconds": time.monotonic()-started})
    output = workflow.ROOT / "reports"
    output.mkdir(exist_ok=True)
    (output / "local_comparison.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    (output / "local_comparison_splits.json").write_text(json.dumps({"development_smiles": frame.standard_smiles.tolist(),
        "splits": manifest}, indent=2), encoding="utf-8")
    oof.to_csv(output / "local_comparison_oof.csv", index=False)
    table = [{"protocol": protocol, "variant": variant, "mse": values["mse"], "rmse": values["rmse"],
        "skill_vs_constant": values["skill_vs_constant"], "repeat_mse_std": values["repeat_mse_std"]}
        for protocol, variants in report["protocols"].items() for variant, values in variants.items()]
    summary = pd.DataFrame(table)
    summary.to_csv(output / "local_comparison.csv", index=False)
    print(summary.to_string(index=False), flush=True)
    print(f"[local score] completed in {report['elapsed_seconds']:.1f}s; no final model overwritten", flush=True)


def main(argv=None):
    from src.pipeline import load_config
    parser = argparse.ArgumentParser(description="Local OOF score comparison; real fits on Mac only")
    parser.add_argument("--config", default="configs/xgb_weighted.json")
    args = parser.parse_args(argv)
    run(load_config(args.config))


if __name__ == "__main__":
    main()
