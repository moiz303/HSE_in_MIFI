"""Признаки из метаданных анализа: клеточная линия, штамм, состав панели.

**Почему это вообще законно.** Запрет организаторов касается колонок `CC50_uM`
и `IC50_uM_VACV` тестового файла — из них целевая переменная выводится
арифметикой. Колонки `strain` и `Culture` описывают условия эксперимента, а не
результат, и в тесте они заполнены для всех 597 строк. Модель по структуре их
не видела ни разу — это было крупное упущение.

**Почему это должно работать.** `SI = CC50 / IC50`, а цитотоксичность
измеряется на конкретной клеточной линии и сильно от неё зависит: одно и то же
соединение на Vero и на MT-4 даёт разные CC50. Штамм влияет на чувствительность
к ингибированию. То есть линия и штамм входят в определение цели напрямую, и
модель, которая их не знает, обязана часть разброса списывать в шум.

**Оговорка для отчёта.** Частично модель будет выучивать не химию, а протокол:
клеточная линия коррелирует с лабораторией и работой. Это допустимо — признак
дан в тесте, — но в выводах это стоит называть своим именем.

Текст в исходных данных грязный: `cells HEL`, `HELA cells`, `Human foreskin
fibroblast`, `BSC40`, а поле `strain` часто содержит и культуру тоже
(`Vaccinia virus (strain Copenhagen). culture Vero`). Поэтому нормализация
идёт по словарю шаблонов, а не по точному совпадению.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

# Клеточные линии. Порядок важен: более специфичные шаблоны идут раньше, иначе
# BSC-40 попадёт в BSC-1, а HeLa в HEL.
CELL_LINES: tuple[tuple[str, str], ...] = (
    ("BSC-40", r"bsc[\s_-]?40"),
    ("BSC-1", r"bsc[\s_-]?1\b"),
    ("HeLa", r"hela"),
    ("HEL", r"\bhel\b"),
    ("HFF", r"\bhff\b|foreskin fibroblast"),
    ("Vero", r"\bvero\b"),
    ("A549", r"\ba549\b"),
    ("PRK", r"\bprk\b"),
    ("MT-4", r"\bmt[\s-]?4\b"),
    ("CEM", r"\bcem\b"),
    ("E6SM", r"\be6sm\b"),
    ("BHK-21", r"bhk[\s-]?21|\bbhk\b"),
    ("CV-1", r"\bcv[\s-]?1\b"),
    ("MDCK", r"\bmdck\b"),
    ("CRFK", r"\bcrfk\b"),
    ("RK-13", r"\brk[\s-]?13\b"),
    ("lung fibroblast", r"lung fibroblast"),
)

# Штаммы вируса вакцинии.
STRAINS: tuple[tuple[str, str], ...] = (
    ("Copenhagen", r"copenhagen"),
    ("WR", r"western reserve|\bwr\b"),
    ("Lister", r"lister"),
    ("IHD", r"\bihd\b"),
    ("Tiantan", r"tiantan"),
    ("Elstree", r"elstree"),
)

# Признаки состава панели: работы, где рядом мерили другие ортопоксвирусы,
# обычно используют один протокол на весь набор.
PANEL_FLAGS: tuple[tuple[str, str], ...] = (
    ("cowpox", r"cowpox|grishak"),
    ("ectromelia", r"ectromelia"),
    ("monkeypox", r"monkeypox"),
    ("multi_virus", r"virus.*virus"),
)

UNKNOWN = "unknown"


def _match(text: object, table: tuple[tuple[str, str], ...]) -> str:
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return UNKNOWN
    lowered = str(text).lower()
    for name, pattern in table:
        if re.search(pattern, lowered):
            return name
    return UNKNOWN


def normalize_cell_line(*texts: object) -> str:
    """Клеточная линия из любого из переданных полей.

    Поля перебираются по порядку: сначала выделенная колонка культуры, затем
    текст штамма, где культура часто тоже упоминается.
    """
    for text in texts:
        found = _match(text, CELL_LINES)
        if found != UNKNOWN:
            return found
    return UNKNOWN


def normalize_strain(*texts: object) -> str:
    for text in texts:
        found = _match(text, STRAINS)
        if found != UNKNOWN:
            return found
    return UNKNOWN


def panel_flags(*texts: object) -> dict[str, int]:
    joined = " ".join(str(text).lower() for text in texts if text is not None)
    return {name: int(bool(re.search(pattern, joined))) for name, pattern in PANEL_FLAGS}


def columns_of(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Текст культуры и штамма из таблицы любого из наших форматов.

    Наша собранная база хранит культуру в `cell_line` для строк соревнования и
    в `cell_line_ic50` для строк ChEMBL; тестовый файл — в `Culture`.
    """
    empty = pd.Series([""] * len(frame), index=frame.index)

    culture = empty.copy()
    for name in ("Culture", "cell_line", "cell_line_ic50", "cell_line_cc50"):
        if name in frame.columns:
            culture = culture.where(culture.str.len() > 0, frame[name].fillna("").astype(str))

    strain = frame["strain"].fillna("").astype(str) if "strain" in frame.columns else empty
    return culture, strain


def describe(frame: pd.DataFrame) -> pd.DataFrame:
    """Нормализованные метаданные одной таблицей — для отчёта и проверок."""
    culture, strain = columns_of(frame)
    rows = []
    for culture_text, strain_text in zip(culture, strain):
        flags = panel_flags(culture_text, strain_text)
        rows.append({
            "cell_line": normalize_cell_line(culture_text, strain_text),
            "strain": normalize_strain(strain_text, culture_text),
            **flags,
        })
    return pd.DataFrame(rows, index=frame.index)


def metadata_matrix(
    frame: pd.DataFrame, use_cells: bool = True, use_strain: bool = True,
    use_flags: bool = True,
) -> np.ndarray:
    """Матрица признаков из метаданных: one-hot по словарям плюс флаги панели.

    Словари фиксированы в этом модуле, поэтому на обучении и на тесте колонки
    совпадают по смыслу и порядку — без этого признаки поехали бы.
    """
    table = describe(frame)
    blocks: list[np.ndarray] = []

    if use_cells:
        names = [name for name, _ in CELL_LINES] + [UNKNOWN]
        blocks.append(
            np.stack([(table["cell_line"] == name).to_numpy(dtype=np.float32)
                      for name in names], axis=1)
        )
    if use_strain:
        names = [name for name, _ in STRAINS] + [UNKNOWN]
        blocks.append(
            np.stack([(table["strain"] == name).to_numpy(dtype=np.float32)
                      for name in names], axis=1)
        )
    if use_flags:
        blocks.append(
            np.stack([table[name].to_numpy(dtype=np.float32) for name, _ in PANEL_FLAGS],
                     axis=1)
        )

    if not blocks:
        return np.zeros((len(frame), 0), dtype=np.float32)
    return np.hstack(blocks)


def width(use_cells: bool = True, use_strain: bool = True, use_flags: bool = True) -> int:
    """Сколько колонок даст matrix при таких настройках."""
    total = 0
    if use_cells:
        total += len(CELL_LINES) + 1
    if use_strain:
        total += len(STRAINS) + 1
    if use_flags:
        total += len(PANEL_FLAGS)
    return total
