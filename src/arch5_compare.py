"""Leakage-controlled source hierarchy: public concentration heads + Kaggle residuals."""
import argparse
import joblib
import json
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from src import splits
from src.local_score import score


def exact_logs(frame, endpoint):
    values = pd.to_numeric(frame[endpoint], errors="coerce").to_numpy(dtype=float, copy=True)
    qualifier = endpoint + "_qualifier"
    if qualifier in frame:
        marked = frame[qualifier].astype("string").fillna("").str.strip().ne("").to_numpy()
        values[marked] = np.nan
    result = np.full(len(values), np.nan)
    mask = np.isfinite(values) & (values > 0)
    result[mask] = np.log10(values[mask])
    return result


def public_training_rows(public, allowed_scaffolds):
    scaffolds = public.standard_smiles.map(splits.murcko_scaffold)
    return public.loc[scaffolds.isin(set(allowed_scaffolds))].copy()


def exact_target(frame, endpoint):
    values = pd.to_numeric(frame[endpoint], errors="coerce").to_numpy(dtype=float, copy=True)
    qualifier = endpoint + "_qualifier"
    if qualifier not in frame:
        raise ValueError(f"Missing {qualifier} for public concentration target")
    marked = frame[qualifier].astype("string").fillna("").str.strip().ne("").to_numpy()
    values[marked | ~np.isfinite(values) | (values <= 0)] = np.nan
    return values


def scaffold_folds(smiles, folds, seed):
    groups = defaultdict(list)
    for index, value in enumerate(smiles):
        groups[splits.murcko_scaffold(value)].append(index)
    if len(groups) < folds:
        raise ValueError("Insufficient independent scaffolds for requested folds")
    items = list(groups.values())
    np.random.default_rng(seed).shuffle(items)
    items.sort(key=len, reverse=True)
    result = {str(index): [] for index in range(folds)}
    for indices in items:
        destination = min(result, key=lambda key: len(result[key]))
        result[destination].extend(indices)
    splits.assert_no_leaks(smiles, result)
    return result


def fit_public(public, matrix, params):
    from src.models.xgb_two_head import XGBTwoHead
    model = XGBTwoHead(params)
    model.fit(matrix, exact_target(public, "IC50"), exact_target(public, "CC50"))
    return model


def fit_kaggle(reference, matrix, params):
    from src.models.xgb_two_head import XGBTwoHead
    model = XGBTwoHead(params)
    model.fit(matrix, pd.to_numeric(reference.IC50, errors="coerce").to_numpy(float),
              pd.to_numeric(reference.CC50, errors="coerce").to_numpy(float))
    return model


def public_oof(reference, public, matrix_by_smiles, params, folds, seed):
    """Get honest source-baseline logs for Kaggle rows, excluding each held scaffold."""
    result = {endpoint: np.full(len(reference), np.nan) for endpoint in ("IC50", "CC50")}
    split = scaffold_folds(reference.standard_smiles.tolist(), folds, seed)
    for name, validation in split.items():
        validation = np.asarray(validation, dtype=int)
        training = np.setdiff1d(np.arange(len(reference)), validation)
        held_smiles = set(reference.iloc[validation].standard_smiles)
        held_scaffolds = {splits.murcko_scaffold(value) for value in held_smiles}
        train_smiles = set(reference.iloc[training].standard_smiles)
        allowed = {splits.murcko_scaffold(value) for value in public.standard_smiles}
        allowed -= held_scaffolds
        rows = public_training_rows(public, allowed)
        if rows.empty:
            raise ValueError("No public measurements remain after scaffold exclusion")
        model = fit_public(rows, np.vstack(rows.standard_smiles.map(matrix_by_smiles)), params)
        logs = model.predict_log_concentrations(np.vstack(reference.iloc[validation].standard_smiles.map(matrix_by_smiles)))
        for endpoint in ("IC50", "CC50"):
            result[endpoint][validation] = logs[endpoint]
    return result


def fit_residuals(features, reference, baseline_logs, alpha=10.0):
    models = {}
    for index, endpoint in enumerate(("IC50", "CC50")):
        target = exact_logs(reference, endpoint) - baseline_logs[endpoint]
        mask = np.isfinite(target)
        if not mask.any():
            raise ValueError(f"No exact Kaggle {endpoint} targets for residual fit")
        models[endpoint] = Ridge(alpha=alpha).fit(features[mask], target[mask])
    return models


def benchmark(frame, reference, public, matrix_by_smiles, params=None, folds=3, seed=20261107,
              residual_alpha=10.0):
    if frame.standard_smiles.duplicated().any():
        raise ValueError("Development rows must have unique standardized structures")
    if public.empty or reference.empty:
        raise ValueError("Public measurements and original Kaggle targets are required")
    if not set(reference.standard_smiles).issubset(set(frame.standard_smiles)):
        raise ValueError("Kaggle reference includes structures outside development data")
    if not set(public.standard_smiles).issubset(set(matrix_by_smiles)):
        raise ValueError("Missing public structure features")
    rows, fold_scores = [], []
    for name, validation in scaffold_folds(frame.standard_smiles.tolist(), folds, seed).items():
        held = frame.iloc[np.asarray(validation, dtype=int)]
        training = frame.drop(index=held.index)
        held_smiles = set(held.standard_smiles)
        train_smiles = set(training.standard_smiles)
        allowed = {splits.murcko_scaffold(value) for value in train_smiles}
        allowed -= {splits.murcko_scaffold(value) for value in held_smiles}
        public_train = public_training_rows(public, allowed)
        if public_train.empty:
            raise ValueError("No public measurements remain after outer scaffold exclusion")
        train_ref = reference[reference.standard_smiles.isin(train_smiles)].reset_index(drop=True)
        test_ref = reference[reference.standard_smiles.isin(held_smiles)].copy()
        inner_logs = public_oof(train_ref, public_train, matrix_by_smiles, params, folds, seed + int(name) + 1)
        x_train = np.vstack(train_ref.standard_smiles.map(matrix_by_smiles))
        residual = fit_residuals(x_train, train_ref, inner_logs, alpha=residual_alpha)
        public_model = fit_public(public_train, np.vstack(public_train.standard_smiles.map(matrix_by_smiles)), params)
        x_test = np.vstack(test_ref.standard_smiles.map(matrix_by_smiles))
        base = public_model.predict_log_concentrations(x_test)
        corrected = np.column_stack([base[endpoint] + residual[endpoint].predict(x_test)
                                     for endpoint in ("IC50", "CC50")])
        prediction = np.power(10.0, corrected[:, 1] - corrected[:, 0])
        kaggle_model = fit_kaggle(train_ref, x_train, params)
        control_prediction = kaggle_model.predict_si(x_test)
        test_ref = test_ref.copy()
        test_ref["prediction"] = prediction
        test_ref["kaggle_only"] = control_prediction
        test_ref["fold"] = name
        rows.append(test_ref)
        fold_scores.append({"fold": name, "n_public_train": len(public_train),
                            "n_kaggle_train": len(train_ref), "n_score": len(test_ref),
                            "hierarchical_mse": score(test_ref.SI, prediction)["mse"],
                            "kaggle_only_mse": score(test_ref.SI, control_prediction)["mse"]})
    oof = pd.concat(rows, ignore_index=True)
    if len(oof) != len(reference) or oof.original_row.nunique() != len(reference):
        raise AssertionError("Each original Kaggle row must be scored once")
    return {"metric": "raw SI MSE on original Kaggle targets", "protocol": "Murcko scaffold CV",
            "folds": folds, "holdout_used": False, "test_endpoints_used": False,
            "n_scored_original_rows": len(oof), "mse": score(oof.SI, oof.prediction)["mse"],
            "kaggle_only_mse": score(oof.SI, oof.kaggle_only)["mse"],
            "residual_alpha": residual_alpha,
            "fold_scores": fold_scores}, oof


def run(config):
    from src import xgb_workflow as workflow
    parts = workflow.load_parts()
    frame = pd.concat([parts["train"], parts["validation"]], ignore_index=True)
    public = pd.read_csv(workflow.ROOT / "data/raw/chembl.csv")
    from src.data.standardize import add_standard_smiles
    public, _ = add_standard_smiles(public)
    public = public[public.source.eq("chembl")].reset_index(drop=True)
    reference = pd.read_csv(workflow.ROOT / "data/processed/kaggle_reference.csv")
    reference = reference[reference.standard_smiles.isin(frame.standard_smiles)].copy()
    all_smiles = list(dict.fromkeys([*frame.standard_smiles, *public.standard_smiles]))
    matrix = workflow.features(pd.DataFrame({"standard_smiles": all_smiles}))
    matrix_by_smiles = dict(zip(all_smiles, matrix))
    settings = config.get("source_hierarchy", {})
    reports, predictions = [], []
    repeats = settings.get("repeats", 3)
    for repeat in range(repeats):
        current, rows = benchmark(frame, reference, public, matrix_by_smiles,
            config["model"].get("params"), settings.get("folds", 3), config.get("seed", 20261106) + repeat,
            settings.get("ridge_alpha", 10.0))
        rows["repeat"] = repeat
        reports.append(current)
        predictions.append(rows)
    oof = pd.concat(predictions, ignore_index=True)
    report = {**reports[0], "repeats": repeats, "repeat_reports": reports,
              "n_scored_original_rows": len(reference), "n_oof_predictions": len(oof),
              "mse": score(oof.SI, oof.prediction)["mse"],
              "kaggle_only_mse": score(oof.SI, oof.kaggle_only)["mse"]}
    # Final refit uses development structures only; evaluation above remains OOF.
    allowed = {splits.murcko_scaffold(value) for value in frame.standard_smiles}
    public_final = public_training_rows(public, allowed)
    inner_logs = public_oof(reference, public_final, matrix_by_smiles,
                            config["model"].get("params"), settings.get("folds", 3), config.get("seed", 20261107))
    residual = fit_residuals(np.vstack(reference.standard_smiles.map(matrix_by_smiles)),
                             reference, inner_logs, alpha=settings.get("ridge_alpha", 10.0))
    public_model = fit_public(public_final, np.vstack(public_final.standard_smiles.map(matrix_by_smiles)),
                               config["model"].get("params"))
    joblib.dump({"public_model": public_model, "residual_heads": residual,
                 "config": config, "holdout_used": False, "training_rows": len(frame),
                 "features": "standardized Morgan 2048 + RDKit13",
                 "prediction": "10**((public_CC50+residual_CC50)-(public_IC50+residual_IC50))"},
                workflow.ROOT / "artifacts/arch5_source_hierarchy.joblib")
    report["final_refit"] = {"development_rows": len(frame), "public_records": len(public_final),
                              "artifact": "artifacts/arch5_source_hierarchy.joblib"}
    report["architecture"] = "public exact log-concentration heads + Ridge residuals trained on nested scaffold OOF"
    output = workflow.ROOT / "reports"
    output.mkdir(exist_ok=True)
    (output / "arch5_source_hierarchy.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    oof.to_csv(output / "arch5_source_hierarchy_oof.csv", index=False)
    print(json.dumps(report, indent=2), flush=True)


def main(argv=None):
    from src.pipeline import load_config
    parser = argparse.ArgumentParser(description="Hierarchical public/Kaggle concentration experiment")
    parser.add_argument("--config", default="configs/xgb_arch5.json")
    args = parser.parse_args(argv)
    run(load_config(args.config))


if __name__ == "__main__":
    main()
