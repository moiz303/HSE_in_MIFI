"""
Задача 4 (TASKA.md): Morgan Fingerprints.

SMILES -> RDKit Mol -> Morgan fingerprint -> fixed-length binary numpy array.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator


def smiles_to_morgan(
    smiles: str,
    radius: int = 2,
    n_bits: int = 2048,
    use_chirality: bool = False,
) -> Optional[np.ndarray]:
    """Return a binary Morgan fingerprint as a (n_bits,) uint8 array, or
    None if the SMILES fails to parse."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=radius, fpSize=n_bits, includeChirality=use_chirality
    )
    fp = generator.GetFingerprint(mol)
    arr = np.zeros((n_bits,), dtype=np.uint8)
    on_bits = list(fp.GetOnBits())
    if on_bits:
        arr[on_bits] = 1
    return arr


def morgan_matrix(
    smiles_list: List[str],
    radius: int = 2,
    n_bits: int = 2048,
    use_chirality: bool = False,
) -> np.ndarray:
    """Build the (n_molecules, n_bits) Morgan fingerprint matrix.

    Raises ValueError naming the offending SMILES if any fail to parse —
    by the time features are generated, splits.py should already have
    dropped unparsable molecules, so a failure here signals a real
    contract problem worth surfacing loudly rather than silently zero-filling.
    """
    rows = []
    bad = []
    for i, smi in enumerate(smiles_list):
        fp = smiles_to_morgan(smi, radius=radius, n_bits=n_bits, use_chirality=use_chirality)
        if fp is None:
            bad.append((i, smi))
            rows.append(np.zeros((n_bits,), dtype=np.uint8))
        else:
            rows.append(fp)
    if bad:
        raise ValueError(f"Unparsable SMILES encountered during Morgan FP generation: {bad}")
    return np.vstack(rows)
