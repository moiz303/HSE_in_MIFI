#!/usr/bin/env python
"""Compatibility entry point; the comparison now uses XGBoost.

Prefer `python compare_xgboost.py`. This old command remains available for
existing scripts and imports.
"""

from compare_xgboost import compare, main


if __name__ == "__main__":
    raise SystemExit(main())
