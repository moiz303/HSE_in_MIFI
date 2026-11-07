"""Measure frozen-MolE Ridge after adding globally safe published CC50 data.

Evaluation structures, raw target rows, folds, and Ridge alphas are held fixed
from the preceding nested OOF run so this reports a data-only comparison.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src import local_compare, splits, xgb_workflow as workflow
from src.data.vacv_supplement import eligible_supplement, load_supplement
from src.local_score import score
from src.mole_compare import EMBEDDINGS, VARIANTS, feature_sets, load_embeddings
from src.models.mole_ridge import RidgeTwoHead


ROOT = Path(__file__).resolve().parents[1]
SUPPLEMENT_EMBEDDINGS = ROOT / "data/interim/mole_vacv_supplement_embeddings_new_dataset.npz"
REPORT = ROOT / "reports/mole_ridge_new_dataset_comparison.json"


def append_eligible_supplement(frame, supplement, held_smiles, protocol, embedding_lookup):
    """Append exact CC50-only observation rows after the specified split guard."""
    eligible = eligible_supplement(supplement, held_smiles, protocol)
    exact_cc50 = eligible.loc[eligible.endpoint.eq("CC50") & eligible.relation.eq("=")].copy()
    if exact_cc50.empty:
        return frame.copy(), exact_cc50, {"rows": 0, "structures": 0, "exact_cc50": 0}
    if exact_cc50.standard_smiles.isin(held_smiles).any():
        raise AssertionError("Held validation structures entered supplementary training rows")
    if protocol == "scaffold":
        held_scaffolds = {splits.murcko_scaffold(s) for s in held_smiles}
        if exact_cc50.standard_smiles.map(splits.murcko_scaffold).isin(held_scaffolds).any():
            raise AssertionError("Held validation scaffold entered supplementary training rows")
    appended = pd.DataFrame({
        "standard_smiles": exact_cc50.standard_smiles.to_numpy(),
        "IC50": np.nan,
        "IC50_qualifier": "",
        "CC50": pd.to_numeric(exact_cc50.value, errors="raise").to_numpy(float),
        "CC50_qualifier": "",
        "source": "literature_vacv_2023",
        "doi": exact_cc50.doi.to_numpy(),
        "target_is_label": False,
        "SI": np.nan,
    })
    if not appended.standard_smiles.isin(embedding_lookup).all():
        raise ValueError("Supplement row is missing a pretrained MolE embedding")
    return (pd.concat((frame, appended), ignore_index=True), exact_cc50,
            {"rows": int(len(appended)), "structures": int(appended.standard_smiles.nunique()),
             "exact_cc50": int(len(appended))})


def load_supplement_features(supplement):
    if not SUPPLEMENT_EMBEDDINGS.exists():
        raise FileNotFoundError(f"Missing {SUPPLEMENT_EMBEDDINGS}; run extraction in the MolE environment")
    with np.load(SUPPLEMENT_EMBEDDINGS, allow_pickle=False) as cache:
        smiles = cache["standard_smiles"].astype(str).tolist()
        embeddings = cache["embeddings"]
    expected = sorted(supplement.standard_smiles.unique().tolist())
    if sorted(smiles) != expected or embeddings.shape != (len(expected), 1000):
        raise ValueError("New-dataset MolE cache must contain exactly the globally safe supplement structures")
    embedding_lookup = dict(zip(smiles, embeddings, strict=True))
    structures = pd.DataFrame({"standard_smiles": smiles})
    morgan = workflow.features(structures)
    morgan_lookup = {smile: morgan[index] for index, smile in enumerate(smiles)}
    feature_lookup = {
        smile: {"mole": embeddings[index], "morgan": morgan[index],
                "fusion": np.hstack((embeddings[index], morgan[index]))}
        for index, smile in enumerate(smiles)
    }
    return embedding_lookup, feature_lookup


def supplemental_matrix(exact_cc50, feature_lookup, variant):
    if exact_cc50.empty:
        return np.empty((0, next(iter(feature_lookup.values()))[variant].shape[0]))
    return np.vstack([feature_lookup[s][variant] for s in exact_cc50.standard_smiles])


def selected_alpha(base_report, protocol, repeat, fold, variant):
    matches = [row["selected_alpha"] for row in base_report["fold_scores"]
               if row.get("variant") == variant and row.get("protocol") == protocol
               and row.get("repeat") == repeat and row.get("fold") == fold]
    if len(matches) != 1:
        raise ValueError(f"Expected one preselected alpha for {protocol}/{repeat}/{fold}/{variant}")
    return float(matches[0])


def run():
    started = time.monotonic()
    parts = workflow.load_parts()
    frame = pd.concat((parts["train"], parts["validation"]), ignore_index=True)
    reference = pd.read_csv(ROOT / "data/processed/kaggle_reference.csv")
    original_embeddings = load_embeddings(frame)
    matrices = feature_sets(frame, original_embeddings)
    supplement = load_supplement(ROOT)
    if len(supplement) != 126 or supplement.standard_smiles.nunique() != 19:
        raise ValueError("Curated supplement no longer matches the audited global-safe counts")
    counts = supplement.endpoint.value_counts().to_dict()
    exact_cc50_count = int((supplement.endpoint.eq("CC50") & supplement.relation.eq("=")).sum())
    if counts != {"EC50": 84, "CC50": 42} or exact_cc50_count != 17:
        raise ValueError(f"Unexpected global-safe supplement counts: {counts}, exact CC50={exact_cc50_count}")
    embedding_lookup, feature_lookup = load_supplement_features(supplement)
    base_report = json.loads((ROOT / "reports/mole_ridge_comparison.json").read_text(encoding="utf-8"))
    base_oof = pd.read_csv(ROOT / "reports/mole_ridge_oof.csv")
    base_splits = json.loads((ROOT / "reports/mole_ridge_splits.json").read_text(encoding="utf-8"))
    expected_smiles = frame.standard_smiles.astype(str).tolist()
    if base_splits["development_smiles"] != expected_smiles:
        raise ValueError("Development split changed since the original MolE OOF run")

    after_rows, fold_reports, generated_splits = [], [], []
    for protocol in ("molecule", "scaffold"):
        for repeat in range(3):
            seed = 20261106 + repeat
            split = local_compare.fold_indices(expected_smiles, protocol, 3, seed)
            generated_splits.append({"protocol": protocol, "repeat": repeat, "seed": seed,
                                     "indices_in_development_pool": split})
            for fold_name, test_indices in split.items():
                test_indices = np.asarray(test_indices, dtype=int)
                train_indices = np.setdiff1d(np.arange(len(frame)), test_indices)
                train_frame, test_frame = frame.iloc[train_indices], frame.iloc[test_indices]
                augmented_train, exact_cc50, added_counts = append_eligible_supplement(
                    train_frame, supplement, test_frame.standard_smiles.tolist(), protocol,
                    embedding_lookup)
                rows = reference[reference.standard_smiles.isin(test_frame.standard_smiles)].copy()
                if len(rows) == 0:
                    raise ValueError("Original evaluation fold lost all reference rows")
                lookup = pd.Series(np.arange(len(test_indices)), index=test_frame.standard_smiles)
                positions = rows.standard_smiles.map(lookup).to_numpy(int)
                rows = rows.rename(columns={"SI": "SI_true"})
                for variant in VARIANTS:
                    alpha = selected_alpha(base_report, protocol, repeat, fold_name, variant)
                    extra_x = supplemental_matrix(exact_cc50, feature_lookup, variant)
                    x_train = np.vstack((matrices[variant][train_indices], extra_x))
                    model = RidgeTwoHead(alpha=alpha).fit(x_train, augmented_train)
                    rows[variant] = model.predict_si(matrices[variant][test_indices])[positions]
                    rows[f"alpha_{variant}"] = alpha
                rows["protocol"], rows["repeat"], rows["fold"] = protocol, repeat, fold_name
                rows["scaffold"] = rows.standard_smiles.map(splits.murcko_scaffold)
                after_rows.append(rows)
                fold_scores = {variant: score(rows.SI_true, rows[variant]) for variant in VARIANTS}
                fold_reports.append({"protocol": protocol, "repeat": repeat, "fold": fold_name,
                                     "original_evaluation_rows": int(len(rows)),
                                     "supplement_training": added_counts,
                                     "scores_after": fold_scores})
                print(f"[MolE new data] {protocol} rep={repeat+1}/3 fold={int(fold_name)+1}/3 "
                      f"addedCC50={added_counts['exact_cc50']} "
                      + json.dumps({name: values["mse"] for name, values in fold_scores.items()}), flush=True)

    if generated_splits != base_splits["splits"]:
        raise AssertionError("New-dataset run changed original 3x3 evaluation folds")
    after_oof = pd.concat(after_rows, ignore_index=True)
    expected_rows = len(reference[reference.standard_smiles.isin(frame.standard_smiles)]) * 3 * 2
    if len(after_oof) != expected_rows or not after_oof.groupby(
            ["protocol", "repeat", "original_row"]).size().eq(1).all():
        raise AssertionError("New supplement changed the original 267-row OOF evaluation coverage")
    if set(after_oof.original_row) != set(base_oof.original_row):
        raise AssertionError("Evaluation rows differ from the preserved baseline OOF rows")
    row_keys = ["protocol", "repeat", "fold", "original_row"]
    before_rows = base_oof[row_keys + ["SI_true"]].sort_values(row_keys).reset_index(drop=True)
    after_rows_for_check = after_oof[row_keys + ["SI_true"]].sort_values(row_keys).reset_index(drop=True)
    for key in row_keys:
        before_rows[key] = before_rows[key].astype(str)
        after_rows_for_check[key] = after_rows_for_check[key].astype(str)
    if not before_rows[row_keys].equals(after_rows_for_check[row_keys]) or not np.array_equal(
            before_rows.SI_true.to_numpy(float), after_rows_for_check.SI_true.to_numpy(float)):
        raise AssertionError("Original OOF row/fold/target join changed after adding supplement data")

    protocols = {}
    for protocol in ("molecule", "scaffold"):
        before = base_oof.loc[base_oof.protocol.eq(protocol)]
        after = after_oof.loc[after_oof.protocol.eq(protocol)]
        protocols[protocol] = {}
        for variant in VARIANTS:
            before_score, after_score = score(before.SI_true, before[variant]), score(after.SI_true, after[variant])
            protocols[protocol][variant] = {
                "before_mse": before_score["mse"], "after_mse": after_score["mse"],
                "delta_mse": after_score["mse"] - before_score["mse"],
                "delta_percent": 100 * (after_score["mse"] - before_score["mse"]) / before_score["mse"],
                "before_repeat_mse": [score(g.SI_true, g[variant])["mse"]
                                      for _, g in before.groupby("repeat")],
                "after_repeat_mse": [score(g.SI_true, g[variant])["mse"]
                                     for _, g in after.groupby("repeat")],
            }
    base_validation = base_report["fixed_validation_train_only"]
    validation = parts["validation"]
    validation_supplement = eligible_supplement(supplement, validation.standard_smiles.tolist(), "scaffold")
    augmented_train, exact_validation_cc50, validation_added_counts = append_eligible_supplement(
        parts["train"], validation_supplement, validation.standard_smiles.tolist(), "scaffold",
        embedding_lookup)
    validation_reference = reference[reference.standard_smiles.isin(validation.standard_smiles)]
    valid_lookup = pd.Series(np.arange(len(validation)), index=validation.standard_smiles)
    valid_positions = validation_reference.standard_smiles.map(valid_lookup).to_numpy(int)
    validation_after = {}
    train_validation_features = feature_sets(
        parts["train"], original_embeddings[:len(parts["train"])])
    valid_validation_features = feature_sets(
        validation, original_embeddings[len(parts["train"]):])
    for variant in VARIANTS:
        alpha = float(base_validation[variant]["selected_alpha"])
        extra_x = supplemental_matrix(exact_validation_cc50, feature_lookup, variant)
        x_train = np.vstack((train_validation_features[variant], extra_x))
        valid_x = valid_validation_features[variant]
        model = RidgeTwoHead(alpha=alpha).fit(x_train, augmented_train)
        validation_after[variant] = {
            "before_mse": float(base_validation[variant]["mse"]),
            "after": score(validation_reference.SI, model.predict_si(valid_x)[valid_positions]),
            "selected_alpha_fixed_from_original_train_only_run": alpha,
        }

    # Refit full development rows plus the globally holdout-safe exact CC50 observations.
    all_dev_supplement = eligible_supplement(supplement, [], "scaffold")
    augmented_dev, exact_final_cc50, final_added_counts = append_eligible_supplement(
        frame, all_dev_supplement, [], "scaffold", embedding_lookup)
    final_features = feature_sets(frame, original_embeddings)
    final_artifacts = ROOT / "artifacts/mole"
    final_artifacts.mkdir(parents=True, exist_ok=True)
    final_alpha = base_report["development_refit"]["models"]
    supplement_feature_rows = {
        variant: supplemental_matrix(exact_final_cc50, feature_lookup, variant)
        for variant in VARIANTS
    }
    for variant in VARIANTS:
        alpha = float(final_alpha[variant]["alpha"])
        x_train = np.vstack((final_features[variant], supplement_feature_rows[variant]))
        model = RidgeTwoHead(alpha=alpha).fit(x_train, augmented_dev)
        joblib.dump({"model": model, "variant": variant, "alpha": alpha,
                     "fit_development_structures": len(frame),
                     "added_exact_cc50_observations": int(len(exact_final_cc50)),
                     "supplement_endpoint_used": "CC50 only; IC50 remains missing",
                     "holdout_used": False},
                    final_artifacts / f"ridge_{variant}_new_dataset.joblib", compress=3)

    raw_sha = hashlib.sha256((ROOT / "data/raw/vacv_2023_reporter_observations.csv").read_bytes()).hexdigest()
    report = {
        "metric": "MSE on the unchanged original 267 competition raw-SI rows",
        "note": "Data-only comparison: original fold-specific alphas were frozen; no outer score, holdout, or test targets selected settings.",
        "holdout_used": False, "test_endpoints_used": False,
        "original_evaluation_fold_row_target_join_unchanged": True,
        "development_structures": int(len(frame)),
        "original_evaluation_rows_per_repeat": int(len(reference[reference.standard_smiles.isin(frame.standard_smiles)])),
        "folds": 3, "repeats": 3, "seeds": [20261106, 20261107, 20261108],
        "supplement": {
            "global_safe_observations": int(len(supplement)),
            "global_safe_structures": int(supplement.standard_smiles.nunique()),
            "global_safe_endpoint_counts": counts,
            "global_safe_exact_cc50_observations": exact_cc50_count,
            "global_safe_exact_cc50_structures": int(supplement.loc[
                supplement.endpoint.eq("CC50") & supplement.relation.eq("="), "standard_smiles"].nunique()),
            "exact_ic50_observations_used": 0,
            "ec50_observations_used": 0,
            "raw_csv_sha256": raw_sha,
            "source_rows_preserved": True,
        },
        "selection": "Use original per-outer-fold nested-CV alpha unchanged to isolate training-data effect.",
        "protocols": protocols,
        "fixed_validation_train_only": {"before": base_validation,
                                        "after": validation_after,
                                        "eligible_supplement_counts": {k: v for k, v in validation_added_counts.items()
                                                                         if k != "embedding_matrix"}},
        "folds_detail": fold_reports,
        "final_refit": {"artifacts": {variant: f"artifacts/mole/ridge_{variant}_new_dataset.joblib"
                                       for variant in VARIANTS},
                        "development_structures": int(len(frame)),
                        "added_exact_cc50_observations": int(len(exact_final_cc50)),
                        "added_exact_cc50_structures": int(exact_final_cc50.standard_smiles.nunique()),
                        "IC50_targets_added": 0, "EC50_targets_added": 0,
                        "holdout_used": False},
        "elapsed_seconds": time.monotonic() - started,
    }
    REPORT.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    after_oof.to_csv(ROOT / "reports/mole_ridge_new_dataset_oof.csv", index=False)
    (ROOT / "reports/mole_ridge_new_dataset_splits.json").write_text(json.dumps(
        {"development_smiles": expected_smiles, "splits": generated_splits}, indent=2), encoding="utf-8")
    summary = [{"protocol": protocol, "variant": variant, **values}
               for protocol, variants in protocols.items() for variant, values in variants.items()]
    pd.DataFrame(summary).to_csv(ROOT / "reports/mole_ridge_new_dataset_comparison.csv", index=False)
    print(pd.DataFrame(summary)[["protocol", "variant", "before_mse", "after_mse", "delta_percent"]].to_string(index=False), flush=True)
    print("fixed validation MSE (train only):", json.dumps({v: x["after"]["mse"] for v, x in validation_after.items()}), flush=True)
    print(f"final refit added {final_added_counts['exact_cc50']} exact CC50 observations; holdout unused", flush=True)
    print(f"completed in {report['elapsed_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    run()
