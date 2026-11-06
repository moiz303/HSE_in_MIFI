"""
Задача 5 (TASKA.md): RDKit 2D physicochemical descriptors.

A fixed, named set of standard descriptors — kept small and interpretable
on purpose (this is the classical baseline, not an exhaustive descriptor
dump) per TASKA.md's example list: MW, LogP, TPSA, HBD, HBA, RotatableBonds,
RingCount, HeavyAtomCount, FractionCSP3.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors

DESCRIPTOR_NAMES = [
    "MolWt",
    "LogP",
    "TPSA",
    "NumHDonors",
    "NumHAcceptors",
    "NumRotatableBonds",
    "RingCount",
    "HeavyAtomCount",
    "FractionCSP3",
]


def smiles_to_descriptors(smiles: str) -> Optional[np.ndarray]:
    """Return the fixed-order descriptor vector for one molecule, or None
    if the SMILES fails to parse."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    values = [
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        rdMolDescriptors.CalcTPSA(mol),
        Lipinski.NumHDonors(mol),
        Lipinski.NumHAcceptors(mol),
        Descriptors.NumRotatableBonds(mol),
        rdMolDescriptors.CalcNumRings(mol),
        mol.GetNumHeavyAtoms(),
        rdMolDescriptors.CalcFractionCSP3(mol),
    ]
    return np.array(values, dtype=np.float64)


def descriptor_matrix(smiles_list: List[str]) -> pd.DataFrame:
    """Build the (n_molecules, n_descriptors) RDKit 2D descriptor matrix
    as a DataFrame with DESCRIPTOR_NAMES columns.

    NaN/Inf handling (Задача 5, п.3): any non-finite descriptor value is
    replaced with NaN, so downstream imputation (fit on train only, see
    features/build.py) has a single consistent "missing" sentinel instead
    of having to special-case inf separately.
    """
    rows = []
    bad = []
    for i, smi in enumerate(smiles_list):
        desc = smiles_to_descriptors(smi)
        if desc is None:
            bad.append((i, smi))
            rows.append(np.full((len(DESCRIPTOR_NAMES),), np.nan))
        else:
            rows.append(desc)
    if bad:
        raise ValueError(f"Unparsable SMILES encountered during descriptor generation: {bad}")
    df = pd.DataFrame(rows, columns=DESCRIPTOR_NAMES)
    df = df.replace([np.inf, -np.inf], np.nan)
    return df
