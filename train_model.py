"""Train CC50/IC50 XGBoost heads on Mac; --help is safe anywhere."""
from __future__ import annotations

import argparse
import time
from src.pipeline import load_config
from src import xgb_workflow


def main(argv=None):
    parser = argparse.ArgumentParser(description="CatBoost / XGBoost CC50/IC50 -> SI (train on Mac)")
    parser.add_argument("--config", default="configs/xgb_hybrid.json")
    parser.add_argument("--predict", help="Optional test CSV for submission after training")
    parser.add_argument("--out", default="reports/submission.csv")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.predict:
        config["submission"]["test_csv"] = args.predict
        config["submission"]["output"] = args.out
    for name in ("train", "evaluate"):
        started = time.monotonic()
        print(f"[stage] {name} starting", flush=True)
        getattr(xgb_workflow, name)(config)
        print(f"[stage] {name} finished in {time.monotonic()-started:.1f}s", flush=True)
    if args.predict:
        xgb_workflow.submission(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
