"""Стандартизация структур и агрегация конфликтующих измерений.

Владелец: A (стадия 1).

Порядок важен: сначала приводим структуру к канону, и только потом считаем
дубликаты. Дедупликация по сырой строке SMILES пропускает соли, заряды и
разные способы записи одной и той же молекулы.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*")

_UNCHARGER = rdMolStandardize.Uncharger()
_LARGEST_FRAGMENT = rdMolStandardize.LargestFragmentChooser()


def standardize_smiles(smiles: str) -> str | None:
    """Канонический SMILES: крупнейший фрагмент, снятая соль, нейтрализация.

    Возвращает None для того, что RDKit не разобрал. Таутомеры здесь
    намеренно не канонизируются: если команда решит их приводить, это делается
    одним переключателем на весь датасет, иначе часть молекул канонизирована,
    а часть нет.
    """
    if smiles is None:
        return None
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    try:
        mol = rdMolStandardize.Cleanup(mol)
        mol = _LARGEST_FRAGMENT.choose(mol)
        mol = _UNCHARGER.uncharge(mol)
    except Exception:
        return None
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def molecular_weight(smiles: str) -> float | None:
    """Молекулярная масса — нужна для перевода µg/mL в µM."""
    mol = Chem.MolFromSmiles(str(smiles)) if smiles else None
    return None if mol is None else float(Descriptors.MolWt(mol))


def add_standard_smiles(frame: pd.DataFrame, column: str = "smiles") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Добавить колонку standard_smiles. Неразобранные строки уходят в брак."""
    work = frame.copy()
    work["standard_smiles"] = [standardize_smiles(value) for value in work[column]]
    rejected = work[work["standard_smiles"].isna()].copy()
    if not rejected.empty:
        rejected["reason"] = "RDKit не разобрал smiles"
    accepted = work[work["standard_smiles"].notna()].reset_index(drop=True)
    return accepted, rejected.reset_index(drop=True)


def aggregate_duplicates(
    frame: pd.DataFrame,
    key: str = "standard_smiles",
    value_columns: tuple[str, ...] = ("IC50", "CC50", "SI"),
) -> pd.DataFrame:
    """Свести измерения одной структуры из разных работ в одну строку.

    Медиана в лог-пространстве: концентрации распределены по порядкам, и
    среднее арифметическое смещается к самому большому значению. Исходные
    значения и DOI сохраняются в колонках *_values и doi_all для аудита.
    """
    if frame.empty:
        return frame.copy()

    rows: list[dict[str, object]] = []
    for structure, group in frame.groupby(key, sort=False):
        row: dict[str, object] = {key: structure, "n_measurements": int(len(group))}

        for column in value_columns:
            values = pd.to_numeric(group[column], errors="coerce").dropna()
            values = values[values > 0]
            if values.empty:
                row[column] = np.nan
                row[f"{column}_spread_log10"] = np.nan
            else:
                logs = np.log10(values.to_numpy(dtype=float))
                row[column] = float(10 ** np.median(logs))
                row[f"{column}_spread_log10"] = float(logs.max() - logs.min())
            row[f"{column}_values"] = ";".join(f"{value:g}" for value in values)

        if "doi" in group.columns:
            dois = sorted({str(value) for value in group["doi"].dropna() if str(value).strip()})
            row["doi"] = dois[0] if dois else ""
            row["doi_all"] = ";".join(dois)

        # Провенанс: из каких источников и статей собрана строка.
        for column in ("source", "source_id", "document_id"):
            if column in group.columns:
                values = sorted({str(value) for value in group[column].dropna() if str(value).strip()})
                row[column] = ";".join(values)

        # Цензура: если хотя бы одно измерение в группе было со знаком, строка
        # помечается. Без этого флага B не сможет замерить вклад таких строк.
        censored = False
        for column in ("IC50_qualifier", "CC50_qualifier"):
            if column in group.columns:
                marks = {str(value).strip() for value in group[column].fillna("")}
                censored = censored or bool(marks - {""})
                row[column] = ";".join(sorted(marks - {""}))
        row["censored"] = censored

        # Артефакт цензуры: SI = 1 только потому, что оба значения упёрлись в
        # одну максимальную концентрацию. Флаг нужен B для отбора обучающей
        # выборки, поэтому он не теряется при агрегации.
        if "si_artifact" in group.columns:
            row["si_artifact"] = bool(group["si_artifact"].fillna(False).astype(bool).any())

        # Цель задана меткой организаторов, а не измерена. Флаг нужен, чтобы
        # обучение могло исключить такие строки и замерить разницу.
        if "target_is_label" in group.columns:
            row["target_is_label"] = bool(
                group["target_is_label"].fillna(False).astype(bool).any()
            )

        first = group.iloc[0]
        for column in (
            "smiles",
            "virus",
            "strain",
            "cell_line",
            "assay",
            "assay_type",
            "cell_line_ic50",
            "cell_line_cc50",
            "cc50_cell_line_match",
        ):
            if column in group.columns:
                row.setdefault(column, first[column])

        rows.append(row)

    return pd.DataFrame(rows)
