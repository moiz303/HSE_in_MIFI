"""Comparable OOF and fixed-validation evaluation for frozen MolE + Ridge."""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src import local_compare, splits, xgb_workflow as workflow
from src.local_score import score
from src.models.mole_ridge import RidgeTwoHead


ROOT = Path(__file__).resolve().parents[1]
EMBEDDINGS = ROOT / "data/interim/mole_development_embeddings.npz"
VARIANTS = ("mole", "morgan", "fusion")
ALPHA_GRID = (100.0, 1000.0, 10000.0)


def load_embeddings(frame: pd.DataFrame, path: Path = EMBEDDINGS) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; run `python -m src.mole_extract` in the MolE environment")
    with np.load(path, allow_pickle=False) as cached:
        cached_smiles = cached["standard_smiles"].astype(str).tolist()
        matrix = cached["embeddings"]
    expected = frame.standard_smiles.astype(str).tolist()
    if cached_smiles != expected or matrix.shape[0] != len(frame) or not np.isfinite(matrix).all():
        raise ValueError("MolE embedding cache does not exactly align with development rows")
    return matrix


def feature_sets(frame: pd.DataFrame, embeddings: np.ndarray) -> dict[str, np.ndarray]:
    morgan = workflow.features(frame)
    return {"mole": embeddings, "morgan": morgan,
            "fusion": np.hstack((embeddings, morgan))}


def select_alpha(frame, reference, matrix, indices, seed, candidates=ALPHA_GRID):
    """Choose Ridge regularization by inner scaffold OOF raw-SI MSE on indices only."""
    indices = np.asarray(indices, dtype=int)
    inner_frame = frame.iloc[indices].reset_index(drop=True)
    inner_reference = reference[reference.standard_smiles.isin(inner_frame.standard_smiles)]
    fold = local_compare.fold_indices(inner_frame.standard_smiles.tolist(), "scaffold", 3, seed)
    predictions = {float(alpha): {"truth": [], "prediction": []} for alpha in candidates}
    for fold_name, valid in fold.items():
        valid = np.asarray(valid, dtype=int)
        train = np.setdiff1d(np.arange(len(inner_frame)), valid)
        valid_frame = inner_frame.iloc[valid]
        rows = inner_reference[inner_reference.standard_smiles.isin(valid_frame.standard_smiles)]
        if rows.empty:
            raise ValueError("An inner scaffold fold has no original competition targets")
        lookup = pd.Series(np.arange(len(valid)), index=valid_frame.standard_smiles)
        positions = rows.standard_smiles.map(lookup).to_numpy(int)
        for alpha in candidates:
            model = RidgeTwoHead(alpha=float(alpha)).fit(matrix[indices[train]], inner_frame.iloc[train])
            predicted = model.predict_si(matrix[indices[valid]])[positions]
            predictions[float(alpha)]["truth"].extend(rows.SI.to_numpy(float))
            predictions[float(alpha)]["prediction"].extend(predicted.tolist())
    scores = {str(alpha): score(values["truth"], values["prediction"])
              for alpha, values in predictions.items()}
    selected = min((float(alpha) for alpha in candidates), key=lambda alpha: (scores[str(alpha)]["mse"], alpha))
    return selected, scores


def benchmark(frame, reference, matrices, folds=3, repeats=3, seed=20261106):
    reference = reference[reference.standard_smiles.isin(frame.standard_smiles)].copy()
    if len(reference) != 267:
        raise ValueError(f"Expected all 267 original competition rows in development, got {len(reference)}")
    all_rows, fold_reports, manifests = [], [], []
    for protocol in ("molecule", "scaffold"):
        for repeat in range(repeats):
            fold = local_compare.fold_indices(frame.standard_smiles.tolist(), protocol, folds, seed + repeat)
            manifests.append({"protocol": protocol, "repeat": repeat, "seed": seed + repeat,
                              "indices_in_development_pool": fold})
            for fold_name, test in fold.items():
                test = np.asarray(test, dtype=int)
                train = np.setdiff1d(np.arange(len(frame)), test)
                train_frame, test_frame = frame.iloc[train], frame.iloc[test]
                rows = reference[reference.standard_smiles.isin(test_frame.standard_smiles)].copy()
                lookup = pd.Series(np.arange(len(test)), index=test_frame.standard_smiles)
                positions = rows.standard_smiles.map(lookup).to_numpy(int)
                rows = rows.rename(columns={"SI": "SI_true"})
                for variant, matrix in matrices.items():
                    alpha, inner_scores = select_alpha(
                        frame, reference, matrix, train, seed + repeat * folds + int(fold_name))
                    model = RidgeTwoHead(alpha=alpha).fit(matrix[train], train_frame)
                    rows[variant] = model.predict_si(matrix[test])[positions]
                    rows[f"alpha_{variant}"] = alpha
                    fold_reports.append({"protocol": protocol, "repeat": repeat, "fold": fold_name,
                                         "variant": variant, "selected_alpha": alpha,
                                         "inner_scaffold_scores": inner_scores})
                rows["constant"] = float(reference[reference.standard_smiles.isin(
                    train_frame.standard_smiles)].SI.mean())
                rows["protocol"], rows["repeat"], rows["fold"] = protocol, repeat, fold_name
                rows["scaffold"] = rows.standard_smiles.map(splits.murcko_scaffold)
                scores = {name: score(rows.SI_true, rows[name]) for name in (*VARIANTS, "constant")}
                fold_reports.append({"protocol": protocol, "repeat": repeat, "fold": fold_name,
                                     "scores": scores})
                all_rows.append(rows)
                print(f"[MolE OOF] {protocol} rep={repeat+1}/{repeats} fold={int(fold_name)+1}/{folds} "
                      + json.dumps({name: value["mse"] for name, value in scores.items()}), flush=True)
    oof = pd.concat(all_rows, ignore_index=True)
    report = {"metric": "mean squared error on original raw SI, lower is better",
              "note": "Development OOF comparison; not an independent public leaderboard score.",
              "development_structures": len(frame), "original_competition_rows": len(reference),
              "folds": folds, "repeats": repeats, "seed": seed, "holdout_used": False,
              "fold_scores": fold_reports, "protocols": {}}
    for protocol, rows in oof.groupby("protocol", sort=False):
        if not rows.groupby("original_row").size().eq(repeats).all():
            raise AssertionError("Each original competition row must be scored once per repeat")
        report["protocols"][protocol] = {}
        for variant in (*VARIANTS, "constant"):
            values = score(rows.SI_true, rows[variant])
            values["repeat_mse"] = [score(part.SI_true, part[variant])["mse"]
                                     for _, part in rows.groupby("repeat")]
            values["strata"] = {}
            is_label = rows.target_is_label.astype("string").str.lower().eq("true")
            for name, mask in {"label": is_label, "measured": ~is_label,
                               "SI_lt_1000": rows.SI_true < 1000,
                               "SI_ge_10000": rows.SI_true >= 10000}.items():
                if mask.any():
                    values["strata"][name] = score(rows.loc[mask, "SI_true"], rows.loc[mask, variant])
            report["protocols"][protocol][variant] = values
    return report, oof, manifests


def fixed_validation(train, validation, reference, matrices, train_matrices, valid_matrices):
    rows = reference[reference.standard_smiles.isin(validation.standard_smiles)].copy()
    lookup = pd.Series(np.arange(len(validation)), index=validation.standard_smiles)
    positions = rows.standard_smiles.map(lookup).to_numpy(int)
    result = {}
    for variant in VARIANTS:
        train_indices = np.arange(len(train))
        alpha, inner_scores = select_alpha(train, reference, train_matrices[variant], train_indices,
                                           seed=20261190 + VARIANTS.index(variant))
        model = RidgeTwoHead(alpha=alpha).fit(train_matrices[variant], train)
        result[variant] = score(rows.SI, model.predict_si(valid_matrices[variant])[positions])
        result[variant]["selected_alpha"] = alpha
        result[variant]["inner_scaffold_scores"] = inner_scores
    result["rows"] = int(len(rows))
    result["train_structures"] = int(len(train))
    return result


def run(folds=3, repeats=3, seed=20261106):
    started = time.monotonic()
    parts = workflow.load_parts()
    frame = pd.concat((parts["train"], parts["validation"]), ignore_index=True)
    reference = pd.read_csv(ROOT / "data/processed/kaggle_reference.csv")
    development_embeddings = load_embeddings(frame)
    matrices = feature_sets(frame, development_embeddings)
    report, oof, manifest = benchmark(frame, reference, matrices, folds, repeats, seed)

    # Preserve the committed validation as a separate train-only report.
    validation_embeddings = development_embeddings[len(parts["train"]):]
    train_matrices = feature_sets(parts["train"], development_embeddings[:len(parts["train"])])
    validation_matrices = feature_sets(parts["validation"], validation_embeddings)
    report["fixed_validation_train_only"] = fixed_validation(
        parts["train"], parts["validation"], reference, matrices, train_matrices, validation_matrices)

    # Fit reproducible development artifacts; the fixed validation is included here.
    final_artifacts = ROOT / "artifacts/mole"
    final_artifacts.mkdir(parents=True, exist_ok=True)
    full_frame_matrices = matrices
    final_alphas = {}
    for variant in VARIANTS:
        selected = [float(item["selected_alpha"]) for item in report["fold_scores"]
                    if item.get("variant") == variant]
        counts = Counter(selected)
        alpha = min((value for value, count in counts.items() if count == max(counts.values())))
        final_alphas[variant] = {"alpha": alpha, "outer_train_selection_counts": dict(sorted(counts.items()))}
        model = RidgeTwoHead(alpha=alpha).fit(full_frame_matrices[variant], frame)
        joblib.dump({"model": model, "variant": variant,
                     "alpha": alpha,
                     "feature_width": full_frame_matrices[variant].shape[1],
                     "fit_structures": len(frame), "holdout_used": False},
                    final_artifacts / f"ridge_{variant}.joblib", compress=3)
    report["development_refit"] = {"fit_structures": len(frame), "holdout_used": False,
                                   "independent_validation_score": None,
                                   "alpha_selection": "modal alpha selected by nested scaffold OOF inside outer training folds",
                                   "models": final_alphas}
    report["elapsed_seconds"] = time.monotonic() - started
    (ROOT / "reports/mole_ridge_comparison.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    oof.to_csv(ROOT / "reports/mole_ridge_oof.csv", index=False)
    (ROOT / "reports/mole_ridge_splits.json").write_text(json.dumps(
        {"development_smiles": frame.standard_smiles.tolist(), "splits": manifest},
        indent=2), encoding="utf-8")
    rows = [{"protocol": protocol, "variant": variant, "mse": values["mse"],
             "rmse": values["rmse"], "repeat_mse_std": float(np.std(values["repeat_mse"]))}
            for protocol, variants in report["protocols"].items()
            for variant, values in variants.items()]
    pd.DataFrame(rows).to_csv(ROOT / "reports/mole_ridge_comparison.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False), flush=True)
    print("Fixed validation, train only:", json.dumps(report["fixed_validation_train_only"]), flush=True)
    print(f"Completed in {report['elapsed_seconds']:.1f}s; no model promotion or holdout scoring.", flush=True)


if __name__ == "__main__":
    run()
