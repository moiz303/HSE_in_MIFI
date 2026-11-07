"""Compare the paired-delta model with and without global-safe VACV observations."""

import argparse
import json

import joblib
import numpy as np
import pandas as pd

from src import splits, xgb_workflow as workflow
from src.data.quality_weights import row_weights
from src.data.vacv_supplement import eligible_supplement, load_supplement
from src.local_compare import fold_indices
from src.local_score import score
from src.models.xgb_pair_delta import XGBPairDelta
from src.models.xgb_two_head import exact_target


SUPPLEMENT_WEIGHT = 0.5
OUTPUT_STEM = "pair_delta_new_dataset"


def append_eligible(original_train, original_weights, supplement, held_smiles, protocol,
                    matrix_by_smiles):
    """Append allowed observations after the original molecule/scaffold split."""
    extra = eligible_supplement(supplement, held_smiles, protocol)
    # EC50 is not IC50; EC50-only and censored CC50 rows add no supervised target.
    exact_cc50 = exact_target(extra, "CC50")
    extra = extra.loc[np.isfinite(exact_cc50)].copy().reset_index(drop=True)
    if protocol == "molecule":
        if set(extra.standard_smiles) & set(held_smiles):
            raise AssertionError("held supplement molecule entered fit rows")
    else:
        held_scaffolds = {splits.murcko_scaffold(s) for s in held_smiles}
        extra_scaffolds = {splits.murcko_scaffold(s) for s in extra.standard_smiles}
        if held_scaffolds & extra_scaffolds:
            raise AssertionError("held supplement scaffold entered fit rows")
    train = pd.concat([original_train, extra], ignore_index=True, sort=False)
    matrix = np.vstack(train.standard_smiles.map(matrix_by_smiles))
    weights = np.concatenate([np.asarray(original_weights, float), np.full(len(extra), SUPPLEMENT_WEIGHT)])
    # EC50 stays a distinct endpoint. The supplement schema must never supply IC50.
    ic50 = exact_target(train, "IC50")
    cc50 = exact_target(train, "CC50")
    if len(extra) and np.isfinite(ic50[len(original_train):]).any():
        raise AssertionError("EC50 or another supplement value was mapped into IC50")
    return train, matrix, ic50, cc50, weights, extra


def benchmark(frame, reference, supplement, quality, matrix_by_smiles, params=None,
              folds=3, repeats=3, seed=20261106, max_references=12, min_similarity=0.35,
              pair_params=None):
    if frame.standard_smiles.duplicated().any() or reference.original_row.duplicated().any():
        raise ValueError("development structures and original-row identifiers must be unique")
    if not set(reference.standard_smiles).issubset(set(frame.standard_smiles)):
        raise ValueError("Kaggle reference includes structures outside development data")
    if len(supplement) != 126 or supplement.standard_smiles.nunique() != 19:
        raise ValueError("Unexpected global-safe supplement data")
    weights = row_weights(frame, quality)
    output, manifests, fold_reports = [], [], []
    for protocol in ("molecule", "scaffold"):
        for repeat in range(repeats):
            folds_by_name = fold_indices(frame.standard_smiles.tolist(), protocol, folds, seed + repeat)
            manifests.append({"protocol": protocol, "repeat": repeat, "seed": seed + repeat,
                              "indices": folds_by_name})
            for fold, held_indices in folds_by_name.items():
                held = np.asarray(held_indices, dtype=int)
                fit = np.setdiff1d(np.arange(len(frame)), held)
                held_smiles = set(frame.iloc[held].standard_smiles)
                fit_smiles = set(frame.iloc[fit].standard_smiles)
                if held_smiles & fit_smiles:
                    raise AssertionError("original fold contains duplicate structures across boundary")
                if protocol == "scaffold":
                    held_scaffolds = {splits.murcko_scaffold(s) for s in held_smiles}
                    fit_scaffolds = {splits.murcko_scaffold(s) for s in fit_smiles}
                    if held_scaffolds & fit_scaffolds:
                        raise AssertionError("original fold contains scaffold leakage")
                original_train = frame.iloc[fit].copy().reset_index(drop=True)
                train, x_train, ic50, cc50, train_weights, extra = append_eligible(
                    original_train, weights[fit], supplement, held_smiles, protocol, matrix_by_smiles)
                score_rows = reference[reference.standard_smiles.isin(held_smiles)].copy()
                if score_rows.empty:
                    raise ValueError("held fold has no original Kaggle rows")
                print(f"[pair-delta supplement] {protocol} repeat={repeat+1}/{repeats} "
                      f"fold={int(fold)+1}/{folds}; fit={len(original_train)}+{len(extra)} "
                      f"score={len(score_rows)}", flush=True)
                model = XGBPairDelta(params, pair_params, max_references=max_references,
                                     min_similarity=min_similarity)
                model.fit(x_train, ic50, cc50, train_weights)
                x_held = np.vstack(frame.iloc[held].standard_smiles.map(matrix_by_smiles))
                baseline = model.baseline.predict_si(x_held)
                prediction = model.predict_si(x_held)
                positions = score_rows.standard_smiles.map(
                    pd.Series(np.arange(len(held)), index=frame.iloc[held].standard_smiles).to_dict()).to_numpy(int)
                score_rows = score_rows.rename(columns={"SI": "SI_true"})
                score_rows["pair_delta_new_dataset"] = prediction[positions]
                score_rows["weighted_new_dataset"] = baseline[positions]
                score_rows["pair_coverage_new_dataset"] = model.last_pair_coverage[positions]
                score_rows["protocol"], score_rows["repeat"], score_rows["fold"] = protocol, repeat, fold
                output.append(score_rows)
                fold_reports.append({"protocol": protocol, "repeat": repeat, "fold": fold,
                    "original_fit_structures": len(original_train),
                    "supplement_fit_observations": len(extra),
                    "supplement_fit_structures": int(extra.standard_smiles.nunique()),
                    "supplement_exact_IC50": int(np.isfinite(ic50[len(original_train):]).sum()),
                    "supplement_exact_CC50": int(np.isfinite(cc50[len(original_train):]).sum()),
                    "pair_rows": model.n_pairs})
    oof = pd.concat(output, ignore_index=True)
    expected = len(reference) * repeats
    counts = oof.groupby(["protocol", "original_row"]).size()
    if len(oof) != expected * 2 or not counts.eq(repeats).all():
        raise AssertionError("Each original Kaggle row must be scored once per repeat and protocol")
    metrics = {}
    for protocol, rows in oof.groupby("protocol", sort=False):
        metrics[protocol] = {name: score(rows.SI_true, rows[column]) for name, column in {
            "pair_delta_new_dataset": "pair_delta_new_dataset",
            "weighted_new_dataset": "weighted_new_dataset"}.items()}
        metrics[protocol]["pair_coverage"] = float(rows.pair_coverage_new_dataset.mean())
    return {"metric": "raw SI MSE on original Kaggle rows; repeated OOF predictions",
            "folds": folds, "repeats": repeats, "seed": seed,
            "holdout_used": False, "test_endpoints_used": False,
            "supplement_global_safe_observations": len(supplement),
            "supplement_global_safe_structures": int(supplement.standard_smiles.nunique()),
            "supplement_endpoints": supplement.endpoint.value_counts().to_dict(),
            "supplement_exact_CC50": int((supplement.endpoint.eq("CC50") & supplement.relation.eq("=")).sum()),
            "supplement_exact_IC50": 0, "supplement_weight": SUPPLEMENT_WEIGHT,
            "protocols": metrics, "fold_training": fold_reports}, oof, manifests


def run(config):
    parts = workflow.load_parts()
    frame = pd.concat([parts["train"], parts["validation"]], ignore_index=True)
    root = workflow.ROOT
    reference = pd.read_csv(root / "data/processed/kaggle_reference.csv")
    reference = reference[reference.standard_smiles.isin(frame.standard_smiles)].copy()
    quality = pd.read_csv(root / "data/processed/public_quality.csv")
    supplement = load_supplement(root)
    options_weight = config.get("pair_delta", {}).get("supplement_weight", SUPPLEMENT_WEIGHT)
    if options_weight != SUPPLEMENT_WEIGHT:
        raise ValueError("Supplement weight is fixed at 0.5; it is not selected by OOF")
    exact_cc50_supplement = supplement.loc[
        supplement.endpoint.eq("CC50") & supplement.relation.eq("=")].copy()
    all_smiles = list(dict.fromkeys([*frame.standard_smiles, *exact_cc50_supplement.standard_smiles]))
    matrix = workflow.features(pd.DataFrame({"standard_smiles": all_smiles}))
    matrix_by_smiles = dict(zip(all_smiles, matrix))
    options = config.get("pair_delta", {})
    report, oof, manifests = benchmark(frame, reference, supplement, quality, matrix_by_smiles,
        config["model"].get("params"), options.get("folds", 3), options.get("repeats", 3),
        config.get("seed", 20261106), options.get("max_references", 12), options.get("min_similarity", .35),
        options.get("pair_params"))

    baseline_path = root / "reports/pair_delta_oof.csv"
    baseline = pd.read_csv(baseline_path)
    if baseline.groupby(["protocol", "original_row"]).size().ne(options.get("repeats", 3)).any():
        raise ValueError("Saved baseline OOF does not match requested repeat count")
    keys = ["protocol", "repeat", "original_row"]
    paired = oof.merge(baseline[keys + ["fold", "SI_true", "pair_delta", "weighted_baseline"]],
                        on=keys, how="left", suffixes=("", "_before"), validate="one_to_one")
    if (paired["pair_delta"].isna().any() or not np.allclose(paired.SI_true, paired.SI_true_before)
            or not paired.fold.astype(str).eq(paired.fold_before.astype(str)).all()):
        raise ValueError("Baseline OOF is missing rows or target values differ")
    report["before_after"] = {}
    for protocol, rows in paired.groupby("protocol", sort=False):
        report["before_after"][protocol] = {
            "pair_delta_before": score(rows.SI_true, rows.pair_delta),
            "pair_delta_after": score(rows.SI_true, rows.pair_delta_new_dataset),
            "weighted_baseline_before": score(rows.SI_true, rows.weighted_baseline),
            "weighted_baseline_after": score(rows.SI_true, rows.weighted_new_dataset)}

    # OOF assessment is complete; only then refit the separate new-data artifact.
    _, x_train, ic50, cc50, train_weights, final_extra = append_eligible(
        frame, row_weights(frame, quality), supplement, set(), "scaffold", matrix_by_smiles)
    model = XGBPairDelta(config["model"].get("params"), options.get("pair_params"),
                         max_references=options.get("max_references", 12),
                         min_similarity=options.get("min_similarity", .35))
    model.fit(x_train, ic50, cc50, train_weights)
    artifact = root / "artifacts/xgb_pair_delta_new_dataset.joblib"
    artifact.parent.mkdir(exist_ok=True)
    metadata = {"model": model, "config": config, "holdout_used": False, "test_endpoints_used": False,
        "development_structures": len(frame), "supplement_observations": len(final_extra),
        "supplement_structures": int(final_extra.standard_smiles.nunique()),
        "total_fit_rows": len(frame) + len(final_extra),
        "exact_training_targets": {"IC50": int(np.isfinite(ic50).sum()),
                                    "CC50": int(np.isfinite(cc50).sum())},
        "supplement_global_safe_observations": len(supplement),
        "supplement_global_safe_structures": int(supplement.standard_smiles.nunique()),
        "supplement_weight": SUPPLEMENT_WEIGHT, "supplement_exact_IC50": 0,
        "supplement_exact_CC50": int(np.isfinite(cc50[len(frame):]).sum()),
        "feature_contract": "Morgan2048 + RDKit 13 descriptors",
        "prediction": "10**(logCC50_pred - logIC50_pred)"}
    joblib.dump(metadata, artifact)
    loaded = joblib.load(artifact)
    sanity = loaded["model"].predict_si(np.vstack(frame.standard_smiles.iloc[:3].map(matrix_by_smiles)))
    if not np.isfinite(sanity).all() or (sanity <= 0).any():
        raise AssertionError("Reloaded final artifact produced invalid development predictions")
    report["final_refit"] = {key: value for key, value in metadata.items() if key != "model"}
    report["final_refit"]["artifact"] = str(artifact.relative_to(root))
    report["final_refit"]["reload_sanity_prediction_count"] = len(sanity)

    reports = root / "reports"
    reports.mkdir(exist_ok=True)
    (reports / f"{OUTPUT_STEM}_comparison.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    (reports / f"{OUTPUT_STEM}_splits.json").write_text(json.dumps(manifests, indent=2), encoding="utf-8")
    oof.to_csv(reports / f"{OUTPUT_STEM}_oof.csv", index=False)
    table = pd.DataFrame([{"protocol": protocol, "method": method, "mse": values["mse"], "n": values["n"]}
        for protocol, variants in report["before_after"].items() for method, values in variants.items()])
    table.to_csv(reports / f"{OUTPUT_STEM}_mse.csv", index=False)
    print(table.to_string(index=False), flush=True)
    print(json.dumps(report["final_refit"], indent=2), flush=True)
    return report


def main(argv=None):
    from src.pipeline import load_config
    parser = argparse.ArgumentParser(description="Leakage-controlled pair-delta supplement OOF comparison")
    parser.add_argument("--config", default="configs/xgb_pair_delta_new_dataset.json")
    args = parser.parse_args(argv)
    run(load_config(args.config))


if __name__ == "__main__":
    main()
