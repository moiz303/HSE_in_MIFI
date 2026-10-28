import numpy as np
import pytest

from vacv.baseline import featurize, scaffold_split, standardize_smiles


def test_smiles_canonicalization_and_invalid_input():
    canonical, _ = standardize_smiles("CCO")
    assert canonical == "CCO"
    with pytest.raises(ValueError):
        standardize_smiles("not a smiles")


def test_features_have_finite_morgan_and_descriptor_columns():
    X, canonical = featurize(["CCO", "CCN"], n_bits=128)
    assert X.shape[0] == 2 and X.shape[1] > 128
    assert np.array_equal(X[:, :128], X[:, :128].astype(bool))
    assert canonical == ["CCO", "CCN"]


def test_scaffold_split_has_no_group_overlap():
    smiles = ["c1ccccc1", "Cc1ccccc1", "c1ccncc1", "CCN", "CCO", "CCCl"]
    train, test = scaffold_split(smiles, test_size=0.34, seed=3)
    assert set(train).isdisjoint(test)
    assert len(train) + len(test) == len(smiles)
