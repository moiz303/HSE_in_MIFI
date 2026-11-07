"""Train CC50/IC50 XGBoost heads on Mac; --help is safe anywhere."""
from __future__ import annotations

import argparse
from src.pipeline import load_config
from src import xgb_workflow


def main(argv=None):
    parser = argparse.ArgumentParser(description="CatBoost / XGBoost CC50/IC50 -> SI (train on Mac)")
    parser.add_argument("--config", default="configs/catboost.json")
    parser.add_argument("--predict", help="Optional test CSV for submission after training")
    parser.add_argument("--out", default="reports/submission.csv")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.predict:
        config["submission"]["test_csv"] = args.predict
        config["submission"]["output"] = args.out
    xgb_workflow.train(config)
    xgb_workflow.evaluate(config)
    if args.predict:
        xgb_workflow.submission(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
