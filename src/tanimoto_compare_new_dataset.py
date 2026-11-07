"""Compare original Tanimoto OOF with fold-safe exact-CC50 VACV augmentation."""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd

from src import xgb_workflow as workflow
from src.data.vacv_supplement import eligible_supplement, load_supplement
from src.features.fingerprints import MORGAN_BITS, morgan_matrix
from src.local_compare import fold_indices
from src.local_score import score
from src.models.tanimoto_krr import TanimotoKRR
from src.models.xgb_two_head import exact_target


def exact_cc50_rows(supplement: pd.DataFrame) -> pd.DataFrame:
    """Retain exact CC50 observations; never reinterpret EC50 as IC50."""
    values = exact_target(supplement, "CC50")
    rows = supplement.loc[np.isfinite(values)].copy().reset_index(drop=True)
    if not rows.empty:
        rows["IC50"] = np.nan
        rows["IC50_qualifier"] = ""
        rows["CC50"] = values[np.isfinite(values)]
        rows["CC50_qualifier"] = ""
        rows["EC50"] = np.nan
    return rows


def run() -> dict:
    parts = workflow.load_parts()
    frame = pd.concat([parts["train"], parts["validation"]], ignore_index=True)
    reference = pd.read_csv(workflow.ROOT / "data/processed/kaggle_reference.csv")
    reference = reference[reference.standard_smiles.isin(frame.standard_smiles)].copy()
    old_oof_path = workflow.ROOT / "reports/tanimoto_oof.csv"
    old_oof = pd.read_csv(old_oof_path)
    old_oof = old_oof[old_oof.standard_smiles.isin(reference.standard_smiles)].copy()
    supplement = load_supplement(workflow.ROOT)
    features = workflow.features(frame)
    fingerprints = features[:, :MORGAN_BITS]
    targets = {name: exact_target(frame, name) for name in ("IC50", "CC50")}
    records = []
    fold_metadata = []

    for protocol in ("molecule", "scaffold"):
        for repeat in range(3):
            fold_map = fold_indices(frame.standard_smiles.tolist(), protocol, 3, 20261106 + repeat)
            for fold, indices in fold_map.items():
                fold_id = int(fold)
                held = np.asarray(indices)
                fit = np.setdiff1d(np.arange(len(frame)), held)
                eligible = eligible_supplement(
                    supplement, frame.iloc[held].standard_smiles.tolist(), protocol
                )
                added = exact_cc50_rows(eligible)
                extra_fp = morgan_matrix(added.standard_smiles.tolist(), radius=2, bits=MORGAN_BITS)
                fit_fp = np.concatenate([fingerprints[fit], extra_fp], axis=0)
                fit_ic50 = np.concatenate([targets["IC50"][fit], np.full(len(added), np.nan)])
                fit_cc50 = np.concatenate([targets["CC50"][fit], added.CC50.to_numpy(float)])
                model = TanimotoKRR({"alpha": 1.0}).fit(fit_fp, fit_ic50, fit_cc50)

                rows = reference[reference.standard_smiles.isin(frame.iloc[held].standard_smiles)].copy()
                lookup = pd.Series(np.arange(len(held)), index=frame.iloc[held].standard_smiles)
                positions = rows.standard_smiles.map(lookup).to_numpy(int)
                rows["prediction"] = model.predict_si(fingerprints[held])[positions]
                rows["protocol"], rows["repeat"], rows["fold"] = protocol, repeat, fold_id
                records.append(rows)
                fold_metadata.append({
                    "protocol": protocol,
                    "repeat": repeat,
                    "fold": fold_id,
                    "original_train_rows": len(fit),
                    "supplement_eligible_observations": len(eligible),
                    "supplement_exact_cc50_rows_added": len(added),
                    "supplement_exact_cc50_structures_added": int(added.standard_smiles.nunique()),
                    "train_rows_after_supplement": len(fit) + len(added),
                    "head_training_counts": model.n_train,
                })
                print(protocol, repeat, fold, len(added), score(rows.SI, rows.prediction)["mse"], flush=True)

    new_oof = pd.concat(records, ignore_index=True)
    if len(new_oof) != len(reference) * 3 * 2 or len(old_oof) != len(reference) * 3 * 2:
        raise AssertionError("original-target OOF coverage changed")
    if set(zip(new_oof["protocol"], new_oof["repeat"], new_oof["fold"], new_oof["standard_smiles"])) != set(
        zip(old_oof["protocol"], old_oof["repeat"], old_oof["fold"], old_oof["standard_smiles"])
    ):
        raise AssertionError("supplement run did not use the original OOF rows/folds")

    table = []
    protocols = {}
    for protocol in ("molecule", "scaffold"):
        before = old_oof.loc[old_oof.protocol.eq(protocol)]
        after = new_oof.loc[new_oof.protocol.eq(protocol)]
        before_mse = score(before.SI, before.prediction)["mse"]
        after_metrics = score(after.SI, after.prediction)
        protocols[protocol] = {
            "n": len(after), "mse_before": before_mse,
            "mse_after": after_metrics["mse"], "delta_mse": after_metrics["mse"] - before_mse,
            "relative_change_pct": 100 * (after_metrics["mse"] / before_mse - 1),
            "after": after_metrics,
        }
        table.append({"protocol": protocol, "n": len(after), "mse_before": before_mse,
                      "mse_after": after_metrics["mse"], "delta_mse": after_metrics["mse"] - before_mse,
                      "relative_change_pct": 100 * (after_metrics["mse"] / before_mse - 1)})

    all_added = exact_cc50_rows(supplement)
    final_fp = morgan_matrix(all_added.standard_smiles.tolist(), radius=2, bits=MORGAN_BITS)
    final_fp = np.concatenate([fingerprints, final_fp], axis=0)
    final_ic50 = np.concatenate([targets["IC50"], np.full(len(all_added), np.nan)])
    final_cc50 = np.concatenate([targets["CC50"], all_added.CC50.to_numpy(float)])
    final_model = TanimotoKRR({"alpha": 1.0}).fit(final_fp, final_ic50, final_cc50)
    artifact = workflow.ROOT / "artifacts/tanimoto_krr_final_new_dataset.joblib"
    joblib.dump({
        "model": final_model,
        "features": "standardized Morgan r2 2048",
        "training_rows": len(frame) + len(all_added),
        "original_training_rows": len(frame),
        "supplement_exact_cc50_rows": len(all_added),
        "supplement_source": "literature_vacv_2023",
        "holdout_used": False,
        "test_data_used": False,
    }, artifact)
    restored = joblib.load(artifact)["model"]
    probe = restored.predict_si(fingerprints[: min(8, len(fingerprints))])
    if not np.isfinite(probe).all():
        raise AssertionError("reloaded final artifact produced invalid predictions")

    report = {
        "architecture": "Tanimoto KRR alpha=1 with exact external CC50 augmentation",
        "folds": 3, "repeats": 3, "original_rows_per_protocol": len(reference),
        "holdout_used": False, "test_data_used": False,
        "supplement": {
            "source": "literature_vacv_2023",
            "global_safe_observations": len(supplement),
            "global_safe_endpoint_counts": supplement.endpoint.value_counts().to_dict(),
            "exact_cc50_rows_used": len(all_added),
            "exact_cc50_structures_used": int(all_added.standard_smiles.nunique()),
            "ec50_used_as_ic50": False,
            "synthetic_pairs_used": False,
        },
        "mse_before_after": protocols,
        "fold_training_metadata": fold_metadata,
        "final_artifact": str(artifact.relative_to(workflow.ROOT)),
        "final_artifact_reload_checked": True,
        "final_head_training_counts": final_model.n_train,
    }
    (workflow.ROOT / "reports/tanimoto_comparison_new_dataset.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    pd.DataFrame(table).to_csv(workflow.ROOT / "reports/tanimoto_mse_before_after_new_dataset.csv", index=False)
    new_oof.to_csv(workflow.ROOT / "reports/tanimoto_oof_new_dataset.csv", index=False)
    print(json.dumps(table, indent=2), flush=True)
    return report


if __name__ == "__main__":
    run()
