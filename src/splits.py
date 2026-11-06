"""Групповое scaffold-разбиение: train / validation / holdout.

Владелец: B (стадия 2).

Случайное разбиение на этой задаче даёт завышенную метрику: один и тот же
химический ряд попадает и в обучение, и в проверку. Группируем по каркасу
Мурко, так что весь ряд целиком уходит в одну часть.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

DEFAULT_FRACTIONS = {"train": 0.7, "validation": 0.15, "holdout": 0.15}
DEFAULT_SEED = 20261106


def murcko_scaffold(smiles: str) -> str:
    """Каркас Мурко. Молекулы без каркаса группируются в бакет "acyclic"."""
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return "invalid"
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    except Exception:
        return "invalid"
    smi = Chem.MolToSmiles(scaffold, canonical=True) if scaffold is not None else ""
    return smi if smi else "acyclic"


def scaffold_split(
    smiles: list[str],
    fractions: dict[str, float] | None = None,
    seed: int = DEFAULT_SEED,
) -> dict[str, list[int]]:
    """Разложить индексы по частям так, чтобы каркас не пересекал границу.

    Крупные группы раскладываются первыми — иначе большой ряд в конце
    перекашивает размеры частей. Порядок групп одного размера фиксирован
    seed'ом, так что разбиение воспроизводится.
    """
    import random

    fractions = dict(fractions or DEFAULT_FRACTIONS)
    total_fraction = sum(fractions.values())
    if abs(total_fraction - 1.0) > 1e-6:
        raise ValueError(f"доли должны давать в сумме 1.0, получено {total_fraction}")

    groups: dict[str, list[int]] = defaultdict(list)
    for index, value in enumerate(smiles):
        groups[murcko_scaffold(value)].append(index)

    ordered = list(groups.items())
    random.Random(seed).shuffle(ordered)
    ordered.sort(key=lambda item: -len(item[1]))

    total = len(smiles)
    targets = {name: fraction * total for name, fraction in fractions.items()}
    result: dict[str, list[int]] = {name: [] for name in fractions}

    for _scaffold, indices in ordered:
        # Кладём группу туда, где больше всего не хватает до целевого размера.
        name = max(result, key=lambda key: targets[key] - len(result[key]))
        result[name].extend(indices)

    return {name: sorted(indices) for name, indices in result.items()}


def assert_no_leaks(smiles: list[str], split: dict[str, list[int]]) -> None:
    """Упасть, если молекула или каркас встречается в двух частях.

    Этот тест стоит в CI на каждом PR: утечка между частями — самая дорогая
    ошибка в задаче, потому что метрика при ней выглядит отлично.
    """
    seen_molecules: dict[str, str] = {}
    seen_scaffolds: dict[str, str] = {}

    for name, indices in split.items():
        for index in indices:
            value = smiles[index]
            if value in seen_molecules and seen_molecules[value] != name:
                raise AssertionError(
                    f"молекула {value} попала и в {seen_molecules[value]}, и в {name}"
                )
            seen_molecules[value] = name

            scaffold = murcko_scaffold(value)
            if scaffold in ("acyclic", "invalid"):
                continue
            if scaffold in seen_scaffolds and seen_scaffolds[scaffold] != name:
                raise AssertionError(
                    f"каркас {scaffold} попал и в {seen_scaffolds[scaffold]}, и в {name}"
                )
            seen_scaffolds[scaffold] = name


def save_manifest(path: str | Path, split: dict[str, list[int]], seed: int, n_rows: int) -> None:
    """Манифест разбиения коммитится: без него результат не воспроизвести."""
    payload = {
        "seed": seed,
        "n_rows": n_rows,
        "sizes": {name: len(indices) for name, indices in split.items()},
        "indices": split,
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_manifest(path: str | Path) -> dict[str, list[int]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return {name: list(indices) for name, indices in payload["indices"].items()}
