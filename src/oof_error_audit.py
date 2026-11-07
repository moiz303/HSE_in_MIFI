"""Development TRAIN/validation OOF error and source audit. Never reads Kaggle test."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from src import xgb_workflow as workflow
from src.data.schema import parse_measurement
from src.models.xgb_two_head import exact_target

ROOT = Path(__file__).resolve().parents[1]


def original_priority_targets(frame, reference):
    """Prefer available exact original competition endpoints; keep independent public fallback."""
    targets = np.column_stack([exact_target(frame, name) for name in ("IC50", "CC50")])
    changes = {}
    for task, name in enumerate(("IC50", "CC50")):
        originals = reference[["standard_smiles", name]].copy()
        parsed = [parse_measurement(value) for value in originals[name]]
        originals["value"] = [value if not qualifier and value is not None else np.nan for value, qualifier in parsed]
        originals = originals[np.isfinite(originals.value) & originals.value.gt(0)].copy()
        originals["log"] = np.log(originals.value)
        mapping = originals.groupby("standard_smiles").log.median()
        values = np.exp(frame.standard_smiles.map(mapping).to_numpy(float))
        mask = np.isfinite(values) & workflow.competition_rows(frame)
        changed = mask & (np.isnan(targets[:, task]) | ~np.isclose(targets[:, task], values, rtol=1e-8, atol=0))
        changes[name] = {"original_available": int(mask.sum()), "changed": int(changed.sum()),
            "unmasked_exact": int((mask & np.isnan(targets[:, task])).sum())}
        targets[mask, task] = values[mask]
    return targets, changes


def run(oof_path=None):
    parts = workflow.load_parts()
    frame = pd.concat([parts["train"], parts["validation"]], ignore_index=True)
    reference = pd.read_csv(ROOT / "data/processed/kaggle_reference.csv")
    reference = reference[reference.standard_smiles.isin(frame.standard_smiles)].copy()
    path = ROOT / (oof_path or "reports/architectures/chemeleon/oof.csv")
    oof = pd.read_csv(path)
    prediction_column = "frozen" if "frozen" in oof else "mlp"
    measured = oof.target_is_label.astype("string").str.lower().eq("false").fillna(False)
    rows = oof.loc[measured].copy()
    rows["squared_error"] = (rows[prediction_column]-rows.SI_true)**2
    rows["absolute_error"] = abs(rows[prediction_column]-rows.SI_true)
    summary = rows.groupby("original_row").agg(
        standard_smiles=("standard_smiles", "first"), SI_true=("SI_true", "first"),
        prediction_mean=(prediction_column, "mean"), mean_squared_error=("squared_error", "mean"),
        max_absolute_error=("absolute_error", "max")).reset_index()
    summary = summary.merge(reference[["original_row", "IC50", "CC50"]], on="original_row", validate="one_to_one")
    fields = ["standard_smiles", "source", "n_measurements", "IC50", "CC50", "IC50_qualifier", "CC50_qualifier",
              "IC50_spread_log10", "CC50_spread_log10", "doi_all", "document_id", "cc50_cell_line_match"]
    source_rows = frame[[name for name in fields if name in frame]].rename(columns={"IC50": "aggregate_IC50", "CC50": "aggregate_CC50"})
    summary = summary.merge(source_rows, on="standard_smiles", validate="many_to_one")
    for name in ("IC50", "CC50"):
        actual = pd.to_numeric(summary[name], errors="coerce")
        aggregate = pd.to_numeric(summary["aggregate_"+name], errors="coerce")
        valid = actual.gt(0) & aggregate.gt(0) & np.isfinite(actual) & np.isfinite(aggregate)
        summary[name+"_aggregate_gap_log10"] = np.where(valid, np.log10(aggregate.where(valid)/actual.where(valid)), np.nan)
        summary[name+"_original_masked_by_aggregate_qualifier"] = actual.gt(0) & summary[name+"_qualifier"].fillna("").ne("")
    summary = summary.sort_values("mean_squared_error", ascending=False).reset_index(drop=True)
    summary["fraction_measured_oof_loss"] = summary.mean_squared_error/summary.mean_squared_error.sum()
    corrected, changes = original_priority_targets(frame, reference)
    baseline = np.column_stack([exact_target(frame, name) for name in ("IC50", "CC50")])
    mixed = frame.source.fillna("").str.contains("kaggle_train") & frame.source.fillna("").str.contains("chembl")
    report = {"purpose": "Retrospective development OOF audit, not independent evaluation", "test_endpoints_used": False,
        "holdout_evaluated": False, "development_rows": len(frame), "original_training_rows": len(reference),
        "measured_original_rows": len(summary), "oof_predictions_per_original_row": rows.groupby("original_row").size().unique().tolist(),
        "measured_pooled_mse": float(rows.squared_error.mean()), "top1_loss_fraction": float(summary.fraction_measured_oof_loss.iloc[:1].sum()),
        "top5_loss_fraction": float(summary.fraction_measured_oof_loss.iloc[:5].sum()),
        "top10_loss_fraction": float(summary.fraction_measured_oof_loss.iloc[:10].sum()),
        "mixed_public_competition_structures": int(mixed.sum()), "original_priority_changes": changes,
        "baseline_exact_targets": dict(zip(("IC50", "CC50"), np.isfinite(baseline).sum(axis=0).tolist())),
        "original_priority_exact_targets": dict(zip(("IC50", "CC50"), np.isfinite(corrected).sum(axis=0).tolist())),
        "rule_to_test": "Exact original competition endpoints take precedence when available; public fallback retained; no row-specific edits, target invention or validation-label changes"}
    directory = ROOT / "reports/finetune"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "error_audit.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    summary.to_csv(directory / "development_error_audit.csv", index=False)
    summary.head(15).to_csv(directory / "largest_development_errors.csv", index=False)
    print(json.dumps(report, indent=2), flush=True)
    print(summary.head(8)[["original_row", "SI_true", "prediction_mean", "fraction_measured_oof_loss", "source",
        "IC50_aggregate_gap_log10", "CC50_aggregate_gap_log10"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oof")
    run(parser.parse_args().oof)
