"""Данные соревнования Kaggle и правило вычисления целевой переменной.

Соревнование: molecular-activity-prediction-hack (Nuclear IT Hack).

Важное свойство этого соревнования: **в тестовом файле даны и CC50_uM, и
IC50_uM_VACV** — оба члена формулы SI = CC50 / IC50. То есть целевая
переменная выводится из признаков арифметически, без модели.

Правило, выведенное из train и воспроизводящее его с MSE 3.4e-08 на всех
320 строках (расхождения — округление SI до трёх знаков в самом файле):

    оба значения есть   -> SI = CC50 / IC50      (237 из 320 строк train)
    IC50 пропущен       -> SI = 10000.0          (81 из 81, маркер неактивности)
    CC50 пропущен       -> SI = 0.0              (2 из 2)

Данные соревнования не коммитятся: их перераспространение — вопрос правил
Kaggle. Пайплайн их скачивает, для этого нужна авторизация
(`kaggle auth login`).
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import zipfile
from pathlib import Path

import pandas as pd

COMPETITION = "molecular-activity-prediction-hack"
KAGGLE_DIR = Path("data/raw/kaggle")

# Колонки в файлах соревнования.
ID_COLUMN = "index"
TARGET_COLUMN = "SI VACV"
SMILES_COLUMN = "SMILES"
CC50_COLUMN = "CC50_uM"
IC50_COLUMN = "IC50_uM_VACV"

# Соглашения организаторов о пропусках, проверенные на train.
INACTIVE_MARKER = 10000.0   # IC50 не определён: соединение неактивно
NO_CC50_VALUE = 0.0         # цитотоксичность не определена


def download(competition: str = COMPETITION, dest: Path | str = KAGGLE_DIR) -> Path:
    """Скачать и распаковать файлы соревнования. Нужна авторизация kaggle."""
    target = Path(dest)
    target.mkdir(parents=True, exist_ok=True)
    archive = target / f"{competition}.zip"

    if not archive.exists():
        result = subprocess.run(
            ["kaggle", "competitions", "download", "-c", competition, "-p", str(target)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise SystemExit(
                "не удалось скачать данные соревнования. Сначала `kaggle auth login`.\n"
                + (result.stderr or result.stdout)
            )

    with zipfile.ZipFile(archive) as bundle:
        bundle.extractall(target)

    checksums = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(target.glob("*.csv"))
    }
    (Path("reports") / "kaggle_files.json").parent.mkdir(parents=True, exist_ok=True)
    (Path("reports") / "kaggle_files.json").write_text(
        json.dumps({"competition": competition, "sha256": checksums}, indent=2),
        encoding="utf-8",
    )
    return target


def load(name: str, directory: Path | str = KAGGLE_DIR) -> pd.DataFrame:
    """Прочитать train.csv, test.csv или sample_submit.csv."""
    path = Path(directory) / f"{name}.csv"
    if not path.exists():
        raise SystemExit(
            f"нет {path}. Скачайте данные: python -m src.pipeline kaggle"
        )
    return pd.read_csv(path)


def selectivity_from_columns(frame: pd.DataFrame) -> pd.Series:
    """Посчитать SI по правилу организаторов.

    Порядок условий важен: строка без CC50 получает 0.0 даже если IC50 задан,
    потому что именно так размечены такие строки в train.
    """
    for column in (CC50_COLUMN, IC50_COLUMN):
        if column not in frame.columns:
            raise ValueError(f"в таблице нет колонки {column!r}")

    cc50 = pd.to_numeric(frame[CC50_COLUMN], errors="coerce")
    ic50 = pd.to_numeric(frame[IC50_COLUMN], errors="coerce")

    selectivity = cc50 / ic50
    selectivity = selectivity.where(ic50.notna(), INACTIVE_MARKER)
    selectivity = selectivity.where(cc50.notna(), NO_CC50_VALUE)
    return selectivity


def verify_rule(train: pd.DataFrame, tolerance: float = 0.01) -> dict:
    """Проверить правило на train. Возвращает сводку для отчёта.

    Это и есть обоснование, почему сабмит считается формулой: не «кажется
    верным», а воспроизводит разметку организаторов на всех строках.
    """
    predicted = selectivity_from_columns(train)
    true = pd.to_numeric(train[TARGET_COLUMN], errors="coerce")
    difference = (predicted - true).abs()
    return {
        "n": int(len(train)),
        "mse": float(((predicted - true) ** 2).mean()),
        "within_tolerance": int((difference < tolerance).sum()),
        "max_abs_error": float(difference.max()),
        "tolerance": tolerance,
    }
