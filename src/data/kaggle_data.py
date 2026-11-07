"""Данные соревнования Kaggle.

Соревнование: molecular-activity-prediction-hack (Nuclear IT Hack).

**Организаторы запретили выводить целевую переменную из колонок тестового
файла.** В test.csv лежат CC50_uM и IC50_uM_VACV, то есть SI = CC50 / IC50
считался бы делением без всякой модели. Этот путь закрыт: функции, считавшие
SI из колонок теста, удалены, и восстанавливать их не нужно. Предсказание
делается только по структуре молекулы.

Уточнение пользователя: отдельная локальная проверка уже сохранённых прогнозов
по тестовым CC50/IC50 разрешена в src/test_diagnostics.py. Этот модуль загрузки,
обучение и инференс не используют полученные диагностические цели. Пропуски не
восстанавливаются условными метками; официальный скрытый score не утверждается.

Запрет касается тестового файла. SI = CC50 / IC50 остаётся законным при сборке
обучающих данных из литературы и баз — см. `src.data.schema.selectivity_index`
и выгрузку ChEMBL.

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

# Колонки, которые НЕЛЬЗЯ использовать для предсказания. Перечислены, чтобы
# проверка ниже могла поймать их появление в пути сабмита.
FORBIDDEN_FEATURE_COLUMNS = ("CC50_uM", "IC50_uM_VACV")


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
    report = Path("reports") / "kaggle_files.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        json.dumps({"competition": competition, "sha256": checksums}, indent=2),
        encoding="utf-8",
    )
    return target


def load(name: str, directory: Path | str = KAGGLE_DIR) -> pd.DataFrame:
    """Прочитать train.csv, test.csv или sample_submit.csv."""
    path = Path(directory) / f"{name}.csv"
    if not path.exists():
        raise SystemExit(f"нет {path}. Скачайте данные: python -m src.pipeline kaggle")
    return pd.read_csv(path)


def training_frame(directory: Path | str = KAGGLE_DIR) -> pd.DataFrame:
    """train.csv соревнования в формате нашего контракта данных.

    Это обучающие данные, в них измеренные CC50 и IC50 использовать можно и
    нужно: запрет касается тестового файла.
    """
    frame = load("train", directory).rename(
        columns={
            SMILES_COLUMN: "smiles",
            "CC50_uM": "CC50",
            "IC50_uM_VACV": "IC50",
            TARGET_COLUMN: "SI",
            "Culture": "cell_line",
        }
    )
    frame["source"] = "kaggle_train"
    frame["virus"] = "vaccinia virus"
    return frame


def assert_no_forbidden_features(frame: pd.DataFrame) -> None:
    """Упасть, если в признаки для предсказания попали запрещённые колонки.

    Страховка от того, что кто-то снова заведёт арифметику: запрет легко
    нарушить случайно, передав в модель весь тестовый фрейм.
    """
    present = [column for column in FORBIDDEN_FEATURE_COLUMNS if column in frame.columns]
    if present:
        raise ValueError(
            "организаторы запретили использовать для предсказания колонки "
            + ", ".join(present)
            + ": предсказание делается только по структуре молекулы"
        )
