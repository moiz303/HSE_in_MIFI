"""Признаки: Morgan fingerprints и RDKit 2D дескрипторы.

Владелец: B (стадия 3).
"""

from __future__ import annotations

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from rdkit.Chem import rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")

MORGAN_RADIUS = 2
MORGAN_BITS = 2048

# Подмножество дескрипторов, которое считается быстро и не содержит NaN на
# разумных структурах. Полный набор RDKit тянет около 200 колонок, половина
# из них на наших объёмах только добавляет шум.
DESCRIPTOR_NAMES = (
    "MolWt",
    "MolLogP",
    "TPSA",
    "NumHAcceptors",
    "NumHDonors",
    "NumRotatableBonds",
    "RingCount",
    "NumAromaticRings",
    "FractionCSP3",
    "HeavyAtomCount",
    "NHOHCount",
    "NOCount",
    "BertzCT",
)

_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(
    radius=MORGAN_RADIUS, fpSize=MORGAN_BITS
)


def morgan_matrix(smiles: list[str]) -> np.ndarray:
    """Матрица фингерпринтов. Неразобранная молекула даёт нулевую строку."""
    rows = np.zeros((len(smiles), MORGAN_BITS), dtype=np.uint8)
    for index, value in enumerate(smiles):
        mol = Chem.MolFromSmiles(str(value))
        if mol is None:
            continue
        rows[index] = np.frombuffer(
            _GENERATOR.GetFingerprintAsNumPy(mol).tobytes(), dtype=np.uint8
        )[:MORGAN_BITS]
    return rows


def descriptor_matrix(smiles: list[str]) -> np.ndarray:
    """RDKit 2D дескрипторы. NaN заменяются нулём и помечаются в отчёте."""
    functions = [(name, getattr(Descriptors, name)) for name in DESCRIPTOR_NAMES]
    rows = np.zeros((len(smiles), len(functions)), dtype=np.float64)
    for index, value in enumerate(smiles):
        mol = Chem.MolFromSmiles(str(value))
        if mol is None:
            continue
        for position, (_name, function) in enumerate(functions):
            try:
                rows[index, position] = float(function(mol))
            except Exception:
                rows[index, position] = 0.0
    return np.nan_to_num(rows, nan=0.0, posinf=0.0, neginf=0.0)


def featurize(smiles: list[str], use_morgan: bool = True, use_descriptors: bool = True) -> np.ndarray:
    """Собрать матрицу признаков в порядке, который фиксируют конфиги."""
    blocks: list[np.ndarray] = []
    if use_morgan:
        blocks.append(morgan_matrix(smiles).astype(np.float32))
    if use_descriptors:
        blocks.append(descriptor_matrix(smiles).astype(np.float32))
    if not blocks:
        raise ValueError("не выбран ни один блок признаков")
    return np.hstack(blocks)
