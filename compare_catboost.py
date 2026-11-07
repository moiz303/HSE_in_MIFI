#!/usr/bin/env python
"""Сравнить CatBoost с текущим HistGradientBoosting на фиксированной validation.

Запускать на машине для обучения:
    python compare_catboost.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from train_model import REPORTS, featurize, load, score
from src.models.flexible import FlexibleSI

ROOT = Path(__file__).resolve().parent
TRAIN_CSV = ROOT / "data" / "processed" / "train.csv"
VALIDATION_CSV = ROOT / "data" / "processed" / "validation.csv"


def compare(
    iterations: int = 400,
    depth: int = 6,
    seed: int = 20261106,
    clip_si: float | None = 20000.0,
) -> dict:
    """Train both learners on train.csv and report metrics on validation.csv."""
    train, validation = load(TRAIN_CSV), load(VALIDATION_CSV)
    print(f"train: {len(train)} rows; validation: {len(validation)} rows", flush=True)
    print("Calculating molecular features...", flush=True)
    features_train = featurize(train)
    features_validation = featurize(validation)

    common = {"transform": "raw", "loss": "squared_error", "clip_si": clip_si}
    settings = {
        "hist_gradient_boosting": {
            **common,
            "learner": "hgb",
            "params": {"max_iter": iterations, "random_state": seed},
        },
        "catboost": {
            **common,
            "learner": "catboost",
            "params": {"iterations": iterations, "depth": depth, "random_seed": seed},
        },
    }

    results = {}
    for name, options in settings.items():
        print(f"Fitting {name}...", flush=True)
        model = FlexibleSI(**options).fit(
            features_train, train["SI"].to_numpy(dtype=float)
        )
        results[name] = score(model, features_validation, validation)
        result = results[name]
        print(
            f"{name}: validation RMSE(log10)={result['rmse_log10']:.4f}; "
            f"competition MSE={result['competition_mse']:.6g}",
            flush=True,
        )
        if "competition_mse_in_distribution" in result:
            print(
                f"  in-distribution competition MSE "
                f"(n={result['n_in_distribution']}): "
                f"{result['competition_mse_in_distribution']:.6g}",
                flush=True,
            )

    metric = "competition_mse_in_distribution"
    if all(metric in result for result in results.values()):
        baseline = results["hist_gradient_boosting"][metric]
        candidate = results["catboost"][metric]
        results["comparison"] = {
            "primary_metric": metric,
            "catboost_relative_change": (
                (candidate - baseline) / baseline if baseline else None
            ),
            "lower_is_better": True,
            "interpretation": (
                "CatBoost is lower on this fixed validation split."
                if candidate < baseline
                else "CatBoost is not lower on this fixed validation split."
            ),
        }
        print(f"Comparison: {results['comparison']['interpretation']}", flush=True)
    else:
        results["comparison"] = {
            "primary_metric": metric,
            "interpretation": (
                "Not available: validation has fewer than 10 competition-source rows."
            ),
        }
        print(results["comparison"]["interpretation"], flush=True)

    return {
        "training_rows": len(train),
        "validation_rows": len(validation),
        "settings": settings,
        "results": results,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--iterations", type=int, default=400)
    parser.add_argument("--depth", type=int, default=6, help="CatBoost tree depth")
    parser.add_argument("--seed", type=int, default=20261106)
    parser.add_argument("--no-clip", action="store_true", help="disable SI prediction cap")
    parser.add_argument(
        "--output",
        type=Path,
        default=REPORTS / "catboost_validation.json",
        help="where to write the comparison report",
    )
    args = parser.parse_args(argv)
    if args.iterations < 1:
        parser.error("--iterations must be positive")
    if args.depth < 1:
        parser.error("--depth must be positive")

    report = compare(
        iterations=args.iterations,
        depth=args.depth,
        seed=args.seed,
        clip_si=None if args.no_clip else 20000.0,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
