#!/usr/bin/env python
"""Константный прогноз — пол, ниже которого модель бессмысленна.

    python constant_baseline.py              # собрать сабмит с константой
    python constant_baseline.py --evaluate   # только замер на валидации

Зачем это в репозитории. Метрика соревнования — MSE по сырому SI, и у
правоскошенной цели с 25% строк-меток на значении 10000 константа «среднее по
обучающим данным» уже даёт очень низкую ошибку. На нашей валидации она
проиграла лучшей модели всего 7%, а двум другим стратегиям выиграла. Без этой
отметки нельзя утверждать, что модель вообще что-то добавляет, поэтому замер
константы входит в отчёт как контрольная точка.

Константа считается **только по обучающим данным**: ни в тест, ни в валидацию
не подглядываем. Для MSE оптимальна именно средняя, а не медиана.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
TRAIN_CSV = ROOT / "data" / "processed" / "train.csv"
VALIDATION_CSV = ROOT / "data" / "processed" / "validation.csv"
CONFIG = ROOT / "configs" / "baseline.json"
REPORTS = ROOT / "reports"


def in_distribution(frame: pd.DataFrame) -> np.ndarray:
    """Строки из train.csv соревнования: только они в распределении теста."""
    if "source" not in frame.columns:
        return np.ones(len(frame), dtype=bool)
    return frame["source"].fillna("").str.contains("kaggle_train").to_numpy()


def constant_from_train(clip_si: float | None = None) -> float:
    frame = pd.read_csv(TRAIN_CSV)
    values = frame["SI"].to_numpy(float)[in_distribution(frame)]
    values = values[np.isfinite(values)]
    if clip_si is not None:
        values = np.minimum(values, clip_si)
    return float(values.mean())


def evaluate(constant: float) -> dict:
    frame = pd.read_csv(VALIDATION_CSV)
    mask = in_distribution(frame)
    true = frame["SI"].to_numpy(float)[mask]
    summary = {
        "константа": constant,
        "строк": int(mask.sum()),
        "mse": float(((constant - true) ** 2).mean()),
        "mse_медианы": float(((np.median(true) - true) ** 2).mean()),
    }
    print(f"константа {constant:.4g} на {summary['строк']} строках валидации: "
          f"MSE {summary['mse']:.4g}")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--evaluate", action="store_true", help="только замер, без файла")
    parser.add_argument("--out", default="reports/submission_constant.csv")
    args = parser.parse_args(argv)

    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    constant = constant_from_train(config["model"].get("clip_si"))
    summary = evaluate(constant)

    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "constant_baseline.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if args.evaluate:
        return 0

    settings = config["submission"]
    test = pd.read_csv(ROOT / settings["test_csv"])
    out = pd.DataFrame({
        settings["id_column"]: test[settings["id_column"]],
        settings["target_column"]: np.full(len(test), constant),
    })
    target = ROOT / args.out
    out.to_csv(target, index=False)
    print(f"{len(out)} строк со значением {constant:.4g} → {target.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
