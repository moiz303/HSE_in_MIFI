"""Conservative provenance weights from ALL raw public records of a structure."""
from pathlib import Path
import numpy as np
import pandas as pd
from src.data.standardize import standardize_smiles

ROOT = Path(__file__).resolve().parents[2]
PUBLIC_SOURCE_FACTOR = 0.5
PAIR_CONFIDENCE = {"vaccinia_same_cell_line": 1.0, "same_cell_line": .75,
                   "vaccinia_other_cell_line": .5, "any_in_document": .25}


def build_quality(raw):
    frame = raw.copy()
    frame["standard_smiles"] = frame.smiles.map(standardize_smiles)
    if frame.standard_smiles.isna().any():
        raise ValueError("Invalid public structure in quality input")
    frame["public_weight"] = PUBLIC_SOURCE_FACTOR * frame.cc50_cell_line_match.map(PAIR_CONFIDENCE).fillna(.25)
    return frame.groupby("standard_smiles", sort=True).agg(
        public_weight=("public_weight", "min"), public_records=("public_weight", "size"),
        pairing_rules=("cc50_cell_line_match", lambda s: ";".join(sorted(set(s.fillna("unknown")))))).reset_index()


def row_weights(frame, quality):
    if quality.standard_smiles.duplicated().any():
        raise ValueError("Duplicate quality keys")
    from_competition = frame.source.fillna("").str.split(";").apply(lambda s: "kaggle_train" in s).to_numpy()
    public = ~from_competition
    if not frame.loc[public, "source"].eq("chembl").all():
        raise ValueError("Unknown training source")
    mapping = quality.set_index("standard_smiles").public_weight
    weights = np.ones(len(frame))
    weights[public] = frame.loc[public, "standard_smiles"].map(mapping).to_numpy(float)
    if not np.isfinite(weights).all() or (weights <= 0).any() or (weights > 1).any():
        raise ValueError("Missing or invalid quality weights; rebuild public_quality.csv")
    return weights


if __name__ == "__main__":
    quality = build_quality(pd.read_csv(ROOT / "data/raw/chembl.csv"))
    path = ROOT / "data/processed/public_quality.csv"
    quality.to_csv(path, index=False)
    print(f"[quality] {len(quality)} structures -> {path}")
