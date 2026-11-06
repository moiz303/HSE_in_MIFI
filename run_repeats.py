"""
Задача 14 (TASKA.md §6): repeat the scaffold split + train + evaluate
cycle on several seeds, to see whether the gap between strategies is
bigger than split-to-split noise. Equivalent to the `./run_mac.sh 5`
entry point TASKA.md mentions; this is the Python-native version.

Usage:
    python run_repeats.py --config configs/baseline.json --seeds 42 43 44 45 46

Writes reports/stability_report.json with per-seed metrics and the
mean/std summary (src.evaluate.repeat_split_stability) for each
(strategy, feature_variant) combination.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from src.evaluate import repeat_split_stability
from src.pipeline import cmd_evaluate, load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.json")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    args = parser.parse_args()

    base_cfg = load_config(args.config)
    per_seed_tables = []

    for seed in args.seeds:
        cfg = copy.deepcopy(base_cfg)
        cfg["seed"] = seed
        # Force a fresh scaffold split for this seed by pointing at seed-
        # specific split artifacts, so repeats don't silently reuse seed
        # 42's train/validation/holdout partition.
        cfg["data"]["train_path"] = f"data/processed/train_seed{seed}.csv"
        cfg["data"]["validation_path"] = f"data/processed/validation_seed{seed}.csv"
        cfg["data"]["splits_manifest"] = f"data/processed/splits_seed{seed}.json"
        cfg["evaluation"]["metrics_output"] = f"reports/metrics_seed{seed}.json"
        result = cmd_evaluate(cfg)
        per_seed_tables.append({"seed": seed, "comparison_table": result["comparison_table"]})

    # Aggregate RMSE across seeds per (strategy, feature_variant).
    by_combo = {}
    for entry in per_seed_tables:
        for row in entry["comparison_table"]:
            key = (row["strategy"], row["feature_variant"])
            by_combo.setdefault(key, []).append(row["rmse"])

    summary = {
        f"{strategy}__{variant}": repeat_split_stability(rmses)
        for (strategy, variant), rmses in by_combo.items()
    }

    out = {"per_seed": per_seed_tables, "stability_summary": summary}
    out_path = Path("reports/stability_report.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print("[run_repeats] stability summary (RMSE across seeds):")
    for combo, stats in summary.items():
        print(f"  {combo}: mean={stats['mean']:.4f} std={stats['std']:.4f} (n={stats['n_repeats']})")
    print(f"[run_repeats] full report written to {out_path}")


if __name__ == "__main__":
    main()
