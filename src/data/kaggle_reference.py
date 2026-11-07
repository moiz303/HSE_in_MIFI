"""Export original competition TRAIN targets, never test concentrations."""
from pathlib import Path
import numpy as np
import pandas as pd
from src.data.kaggle_data import training_frame
from src.data.standardize import standardize_smiles

ROOT = Path(__file__).resolve().parents[2]


def export_reference():
    frame = training_frame(ROOT / "data/raw/kaggle")
    frame["standard_smiles"] = frame.smiles.map(standardize_smiles)
    frame["original_row"] = np.arange(len(frame))
    frame["target_is_label"] = frame.IC50.isna()
    output = ROOT / "data/processed/kaggle_reference.csv"
    frame[["original_row", "standard_smiles", "IC50", "CC50", "SI", "target_is_label"]].to_csv(output, index=False)
    print(f"[reference] {len(frame)} original training rows -> {output}")


def restore_targets(frame, reference):
    """Competition targets take precedence over mixed-source aggregates."""
    result = frame.copy()
    reference = reference[reference.standard_smiles.isin(frame.standard_smiles)]
    grouped = reference.groupby("standard_smiles", sort=False)
    for target in ("IC50", "CC50", "SI"):
        result[target] = pd.to_numeric(result[target], errors="coerce").astype(float)
        values = grouped[target].apply(lambda v: 10 ** np.log10(v[v > 0]).median())
        present = result.standard_smiles.isin(values.index)
        result.loc[present, target] = result.loc[present, "standard_smiles"].map(values)
        if target != "SI":
            result[target + "_qualifier"] = result[target + "_qualifier"].astype("string")
            result.loc[present, target + "_qualifier"] = ""
    flags = grouped.target_is_label.agg(lambda values: values.iloc[0] if values.nunique() == 1 else None)
    present = result.standard_smiles.isin(flags.index)
    result["target_is_label"] = result["target_is_label"].astype(object)
    result.loc[present, "target_is_label"] = result.loc[present, "standard_smiles"].map(flags)
    return result


def original_pairs(frame, predicted, reference):
    mapping = pd.Series(np.asarray(predicted), index=frame.standard_smiles)
    rows = reference[reference.standard_smiles.isin(mapping.index)]
    truth = rows.SI.to_numpy(float)
    forecast = rows.standard_smiles.map(mapping).to_numpy(float)
    valid = np.isfinite(truth) & (truth >= 0)
    return truth[valid], forecast[valid]


if __name__ == "__main__":
    export_reference()
