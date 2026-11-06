"""Контракт данных: колонки, единицы, цензурированные значения, валидация.

Владелец: A (стадия 0). Этот модуль задаёт формат, на который опираются все
остальные треки, поэтому меняется только с оповещением команды.

Итоговый формат, который требует задача:
    smiles, IC50, CC50, SI, doi
Плюс служебные колонки, без которых измерение нельзя интерпретировать.
"""

from __future__ import annotations

import math
import re

import numpy as np
import pandas as pd

# Колонки, которые обязаны быть в итоговой таблице.
REQUIRED_COLUMNS = ("smiles", "IC50", "CC50", "SI")

# Желательные колонки. Отсутствие не отбраковывает строку, но попадает в отчёт.
OPTIONAL_COLUMNS = (
    "doi",
    "source",          # chembl | pubchem | bindingdb | starter | literature
    "source_id",       # идентификатор записи в источнике
    "virus",           # vaccinia virus
    "strain",          # VACV / VV / WR / Copenhagen ...
    "cell_line",       # Vero, HFF, BSC-1 ...
    "assay",           # CPE reduction, plaque reduction ...
    "IC50_qualifier",  # "", ">", "<", ">=", "<="
    "CC50_qualifier",
)

# Все концентрации внутри пайплайна храним в микромолях.
CANONICAL_UNIT = "uM"

# Правдоподобный диапазон концентраций в µM. За его пределами измерение
# почти наверняка означает ошибку единиц в источнике.
CONCENTRATION_RANGE_UM = (1e-6, 1e6)

_QUALIFIER_RE = re.compile(r"^\s*(>=|<=|>|<|~|=)?\s*([0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\s*$")

_UNIT_FACTORS_TO_UM = {
    "m": 1e6,
    "mm": 1e3,
    "um": 1.0,
    "µm": 1.0,
    "μm": 1.0,
    "nm": 1e-3,
    "pm": 1e-6,
}

_MASS_UNITS = {
    "mg/ml": 1e3,   # µg/mL
    "ug/ml": 1.0,
    "µg/ml": 1.0,
    "μg/ml": 1.0,
    "ng/ml": 1e-3,
}


class UnitConversionError(ValueError):
    """Единицы не удалось привести к µM."""


def parse_measurement(raw: object) -> tuple[float | None, str]:
    """Разобрать значение вида "100", ">100", "<0.1", "~5".

    Возвращает (значение, квалификатор). Квалификатор "" означает точное
    измерение. Цензурированные значения мы не выбрасываем молча: они
    возвращаются как число плюс знак, а решение о них принимает очистка.
    """
    if raw is None:
        return None, ""
    if isinstance(raw, (int, float, np.floating, np.integer)):
        value = float(raw)
        return (None, "") if math.isnan(value) else (value, "")

    match = _QUALIFIER_RE.match(str(raw))
    if match is None:
        return None, ""
    qualifier = match.group(1) or ""
    if qualifier == "=":
        qualifier = ""
    return float(match.group(2)), qualifier


def to_micromolar(value: float, unit: str, molecular_weight: float | None = None) -> float:
    """Привести концентрацию к µM.

    Массовые единицы (µg/mL и родственные) требуют молекулярной массы:
    это конверсия со своим отказом, а не умножение на константу. Без массы
    поднимаем UnitConversionError, чтобы строка попала в журнал исключений,
    а не в обучающую выборку с неверным порядком.
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        raise UnitConversionError("пустое значение концентрации")

    key = str(unit).strip().lower().replace(" ", "")
    if key in _UNIT_FACTORS_TO_UM:
        return float(value) * _UNIT_FACTORS_TO_UM[key]

    if key in _MASS_UNITS:
        if molecular_weight is None or not molecular_weight > 0:
            raise UnitConversionError(
                f"для единиц {unit} нужна молекулярная масса"
            )
        ug_per_ml = float(value) * _MASS_UNITS[key]
        # µg/mL → µM: (µg/mL) / (г/моль) * 1000
        return ug_per_ml / float(molecular_weight) * 1e3

    raise UnitConversionError(f"неизвестные единицы: {unit!r}")


def selectivity_index(ic50_um: float, cc50_um: float) -> float:
    """SI = CC50 / IC50, обе концентрации в µM."""
    if not ic50_um > 0:
        raise ValueError("IC50 должен быть положительным")
    return float(cc50_um) / float(ic50_um)


def validate_table(frame: pd.DataFrame, si_tolerance: float = 0.25) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Проверить таблицу по контракту.

    Возвращает (принятые строки, отбракованные строки с колонкой reason).
    Исходный объект не изменяется: очистка пишет производные файлы, а сырую
    таблицу мы не перезаписываем.
    """
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"в таблице нет обязательных колонок: {', '.join(missing)}")

    work = frame.copy()
    reasons: list[str | None] = []
    low, high = CONCENTRATION_RANGE_UM

    for row in work.itertuples(index=False):
        reason: str | None = None
        smiles = getattr(row, "smiles", None)

        if smiles is None or not str(smiles).strip():
            reason = "пустой smiles"
        else:
            numbers: dict[str, float] = {}
            for column in ("IC50", "CC50", "SI"):
                value, _ = parse_measurement(getattr(row, column, None))
                if value is None:
                    reason = f"{column}: нечисловое или пустое значение"
                    break
                numbers[column] = value

            if reason is None:
                for column in ("IC50", "CC50"):
                    if not numbers[column] > 0:
                        reason = f"{column}: не положительное значение"
                        break
                    if not low <= numbers[column] <= high:
                        reason = f"{column}: вне диапазона {low}–{high} µM, похоже на ошибку единиц"
                        break

            if reason is None and not numbers["SI"] > 0:
                reason = "SI: не положительное значение"

            if reason is None:
                expected = selectivity_index(numbers["IC50"], numbers["CC50"])
                if expected > 0 and numbers["SI"] > 0:
                    deviation = abs(math.log10(numbers["SI"]) - math.log10(expected))
                    if deviation > si_tolerance:
                        reason = "SI не согласуется с CC50 / IC50"

        reasons.append(reason)

    work["reason"] = reasons
    accepted = work[work["reason"].isna()].drop(columns=["reason"]).reset_index(drop=True)
    rejected = work[work["reason"].notna()].reset_index(drop=True)
    return accepted, rejected
