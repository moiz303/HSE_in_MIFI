"""
Задача 6 (TASKA.md): assemble the three feature-space variants
(A: Morgan, B: RDKit 2D, C: Morgan + RDKit 2D) with a single rule that
must never be violated: every transform (imputation, scaling, selection)
is FIT on train only and APPLIED (not re-fit) to validation/holdout.

This is why FeatureBuilder is stateful: call .fit(train_smiles) once,
then .transform(any_smiles) for train, validation, and holdout alike.
Never call .fit() on validation/holdout, and never fit on the full
dataset before splitting.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from .descriptors import DESCRIPTOR_NAMES, descriptor_matrix
from .morgan import morgan_matrix

FeatureVariant = Literal["morgan", "rdkit2d", "morgan_rdkit2d"]


@dataclass
class FeatureBuilder:
    variant: FeatureVariant
    morgan_radius: int = 2
    morgan_n_bits: int = 2048
    morgan_use_chirality: bool = False

    def __post_init__(self):
        if self.variant not in ("morgan", "rdkit2d", "morgan_rdkit2d"):
            raise ValueError(f"Unknown feature variant: {self.variant}")
        self._imputer: SimpleImputer | None = None
        self._scaler: StandardScaler | None = None
        self._is_fit = False

    def _morgan(self, smiles_list: List[str]) -> np.ndarray:
        return morgan_matrix(
            smiles_list,
            radius=self.morgan_radius,
            n_bits=self.morgan_n_bits,
            use_chirality=self.morgan_use_chirality,
        )

    def fit(self, train_smiles: List[str]) -> "FeatureBuilder":
        """Fit imputation/scaling for the RDKit-2D block on TRAIN ONLY.
        Morgan fingerprints need no fitting (they're deterministic binary
        hashes), so this only matters for variants that include rdkit2d.
        """
        if self.variant in ("rdkit2d", "morgan_rdkit2d"):
            desc_df = descriptor_matrix(train_smiles)
            self._imputer = SimpleImputer(strategy="median")
            imputed = self._imputer.fit_transform(desc_df.values)
            self._scaler = StandardScaler()
            self._scaler.fit(imputed)
        self._is_fit = True
        return self

    def transform(self, smiles_list: List[str]) -> np.ndarray:
        if not self._is_fit:
            raise RuntimeError("FeatureBuilder.transform() called before .fit(). "
                                "Fit on train only, then transform train/validation/holdout.")
        blocks = []
        if self.variant in ("morgan", "morgan_rdkit2d"):
            blocks.append(self._morgan(smiles_list).astype(np.float64))
        if self.variant in ("rdkit2d", "morgan_rdkit2d"):
            desc_df = descriptor_matrix(smiles_list)
            imputed = self._imputer.transform(desc_df.values)
            scaled = self._scaler.transform(imputed)
            blocks.append(scaled)
        return np.hstack(blocks)

    def fit_transform(self, train_smiles: List[str]) -> np.ndarray:
        return self.fit(train_smiles).transform(train_smiles)

    @property
    def feature_names(self) -> List[str]:
        names = []
        if self.variant in ("morgan", "morgan_rdkit2d"):
            names += [f"morgan_{i}" for i in range(self.morgan_n_bits)]
        if self.variant in ("rdkit2d", "morgan_rdkit2d"):
            names += list(DESCRIPTOR_NAMES)
        return names


def build_feature_matrix(
    variant: FeatureVariant,
    train_smiles: List[str],
    other_smiles: List[str] | None = None,
    **morgan_kwargs,
) -> tuple[np.ndarray, np.ndarray | None, FeatureBuilder]:
    """Convenience wrapper: fit on train_smiles, transform train (and
    optionally another split, e.g. validation) with the SAME fitted builder.
    Returns (train_features, other_features_or_None, fitted_builder).
    """
    builder = FeatureBuilder(variant=variant, **morgan_kwargs)
    train_X = builder.fit_transform(train_smiles)
    other_X = builder.transform(other_smiles) if other_smiles is not None else None
    return train_X, other_X, builder
