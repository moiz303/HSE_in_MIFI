"""Признаки: Morgan fingerprints, MACCS keys и RDKit 2D дескрипторы.

Владелец: B (стадия 3).

Набор признаков — такой же перебираемый параметр, как функция потерь. Пресеты
собраны в PRESETS и перебираются в sweep.py: 41% ошибки метрики приходится на
класс «неактивно», а это задача классификации, где состав признаков обычно
важнее вида потерь.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, MACCSkeys, rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")

MORGAN_RADIUS = 2
MORGAN_BITS = 2048

# Небольшой набор: считается быстро и не даёт NaN на разумных структурах.
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

# Полный набор RDKit — около 200 колонок. Часть из них на наших объёмах добавит
# шум, но это проверяется замером, а не предполагается.
_ALL_DESCRIPTORS = tuple(name for name, _ in Descriptors.descList)

PRESETS: dict[str, dict] = {
    "morgan2048+desc13": {},
    "morgan1024+desc13": {"bits": 1024},
    "morgan4096+desc13": {"bits": 4096},
    "morgan2048r3+desc13": {"radius": 3},
    "morgan2048counts+desc13": {"counts": True},
    "morgan2048+descfull": {"descriptors": "full"},
    "morgan2048+maccs+desc13": {"maccs": True},
    "maccs+descfull": {"morgan": False, "maccs": True, "descriptors": "full"},
    "descfull": {"morgan": False, "descriptors": "full"},
}


@lru_cache(maxsize=16)
def _generator(radius: int, bits: int):
    return rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=bits)


def _mols(smiles: list[str]) -> list:
    return [Chem.MolFromSmiles(str(value)) for value in smiles]


def morgan_matrix(
    smiles: list[str],
    radius: int = MORGAN_RADIUS,
    bits: int = MORGAN_BITS,
    counts: bool = False,
) -> np.ndarray:
    """Матрица фингерпринтов. Неразобранная молекула даёт нулевую строку."""
    generator = _generator(radius, bits)
    dtype = np.uint32 if counts else np.uint8
    rows = np.zeros((len(smiles), bits), dtype=dtype)
    for index, mol in enumerate(_mols(smiles)):
        if mol is None:
            continue
        vector = (
            generator.GetCountFingerprintAsNumPy(mol)
            if counts
            else generator.GetFingerprintAsNumPy(mol)
        )
        rows[index] = np.asarray(vector, dtype=dtype)[:bits]
    return rows


def maccs_matrix(smiles: list[str]) -> np.ndarray:
    """MACCS keys: 167 заранее заданных структурных признаков."""
    rows = np.zeros((len(smiles), 167), dtype=np.uint8)
    for index, mol in enumerate(_mols(smiles)):
        if mol is None:
            continue
        rows[index] = np.frombuffer(
            MACCSkeys.GenMACCSKeys(mol).ToBitString().encode(), dtype=np.uint8
        )[:167] - ord("0")
    return rows


def descriptor_matrix(smiles: list[str], which: str = "small") -> np.ndarray:
    """RDKit 2D дескрипторы. NaN и бесконечности заменяются нулём."""
    names = DESCRIPTOR_NAMES if which == "small" else _ALL_DESCRIPTORS
    functions = [getattr(Descriptors, name) for name in names]
    rows = np.zeros((len(smiles), len(functions)), dtype=np.float64)
    for index, mol in enumerate(_mols(smiles)):
        if mol is None:
            continue
        for position, function in enumerate(functions):
            try:
                rows[index, position] = float(function(mol))
            except Exception:
                rows[index, position] = 0.0
    rows = np.nan_to_num(rows, nan=0.0, posinf=0.0, neginf=0.0)
    # Часть дескрипторов RDKit (например Ipc) даёт значения порядка 1e40 и
    # больше: при приведении к float32 они переполняются в бесконечность, и
    # обучение идёт по мусору. Поэтому зажимаем в диапазон float32 здесь, а не
    # надеемся на nan_to_num после приведения.
    limit = float(np.finfo(np.float32).max) / 16.0
    return np.clip(rows, -limit, limit)


def featurize(
    smiles: list[str],
    use_morgan: bool = True,
    use_descriptors: bool = True,
    morgan: bool | None = None,
    radius: int = MORGAN_RADIUS,
    bits: int = MORGAN_BITS,
    counts: bool = False,
    maccs: bool = False,
    descriptors: str | bool = "small",
) -> np.ndarray:
    """Собрать матрицу признаков.

    use_morgan и use_descriptors оставлены для старых вызовов из конфигов;
    morgan и descriptors — новые имена с теми же значениями.
    """
    with_morgan = use_morgan if morgan is None else morgan
    if descriptors is False:
        which = None
    elif descriptors is True:
        which = "small"
    else:
        which = descriptors if use_descriptors else None

    blocks: list[np.ndarray] = []
    if with_morgan:
        blocks.append(morgan_matrix(smiles, radius, bits, counts).astype(np.float32))
    if maccs:
        blocks.append(maccs_matrix(smiles).astype(np.float32))
    if which is not None:
        blocks.append(descriptor_matrix(smiles, which).astype(np.float32))
    if not blocks:
        raise ValueError("не выбран ни один блок признаков")
    return np.hstack(blocks)
