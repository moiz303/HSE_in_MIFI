"""Leakage-aware Morgan + RDKit baseline for VACV compound classification."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit


def standardize_smiles(smiles: str) -> tuple[str, Chem.Mol]:
    """Parse and canonicalize a SMILES string; raise ValueError if invalid."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")
    canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    return canonical, Chem.MolFromSmiles(canonical)


def featurize(smiles: Sequence[str], n_bits: int = 2048) -> tuple[np.ndarray, list[str]]:
    """Return Morgan radius-2 bits concatenated with fixed RDKit 2D descriptors."""
    rows, canonical = [], []
    for text in smiles:
        smi, mol = standardize_smiles(text)
        canonical.append(smi)
        fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, nBits=n_bits)
        bits = np.zeros((n_bits,), dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(fp, bits)
        desc = np.asarray([fn(mol) for _, fn in Descriptors.descList], dtype=float)
        desc[~np.isfinite(desc)] = np.nan
        rows.append(np.concatenate((bits.astype(float), desc)))
    return np.vstack(rows), canonical


def scaffold_split(smiles: Sequence[str], test_size: float = 0.2, seed: int = 42):
    """Split by Bemis-Murcko scaffold; identical scaffolds stay in one partition."""
    groups = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise ValueError(f"Invalid canonical SMILES: {smi!r}")
        groups.append(MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=True))
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train, test = next(splitter.split(np.zeros(len(groups)), groups=groups))
    return train, test


def fit_baseline(smiles: Sequence[str], labels: Sequence[int], seed: int = 42):
    """Fit RF on scaffold split and return model, split, and untouched-test metrics."""
    X, canonical = featurize(smiles)
    y = np.asarray(labels, dtype=int)
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("labels must contain both classes 0 and 1")
    train, test = scaffold_split(canonical, seed=seed)
    model = RandomForestClassifier(n_estimators=500, class_weight="balanced", random_state=seed,
                                   n_jobs=-1, min_samples_leaf=2)
    model.fit(X[train], y[train])
    prob = model.predict_proba(X[test])[:, 1]
    pred = (prob >= 0.5).astype(int)
    metrics = {"pr_auc": float(average_precision_score(y[test], prob)),
               "roc_auc": float(roc_auc_score(y[test], prob)),
               "mcc_at_0_5": float(matthews_corrcoef(y[test], pred))}
    return model, train, test, metrics
