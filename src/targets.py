"""
Target construction for the VACV SI baseline: SI = CC50 / IC50,
trained/evaluated in log10 space (TASKA.md §2 and §4).

Also implements an EXPLICIT, documented policy for censored IC50/CC50
values (e.g. ">100", "<0.1"), per TASKA.md §4's explicit warning:
censored values must never be silently cast to exact numbers.

Policy (must be agreed with A if the real data's censoring convention
differs — this assumes Pandas-safe prefix strings ">"/"<" in the censor
columns named in configs/baseline.json, with the numeric column holding
the boundary value):
  - A right-censored IC50 (">X", true IC50 unknown but >= X) means the
    true SI is <= CC50/X: we only know an UPPER BOUND on SI for that
    molecule. We EXCLUDE such rows from training and from validation
    scoring by default (policy="exclude"), since training against a
    bound as if it were an exact label would bias the baseline.
  - policy="clip" is provided as an explicit opt-in alternative that
    uses the boundary value as a point estimate (clearly a biased
    approximation) — only to be used if the team agrees a hard
    requirement for 100% sample retention outweighs that bias.
  - Uncensored rows are always used as-is.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

import numpy as np
import pandas as pd

CensorPolicy = Literal["exclude", "clip"]


@dataclass
class TargetConfig:
    ic50_col: str
    cc50_col: str
    censor_ic50_col: str | None = None
    censor_cc50_col: str | None = None
    policy: CensorPolicy = "exclude"


def _row_is_censored(df: pd.DataFrame, col: str | None) -> pd.Series:
    if col is None or col not in df.columns:
        return pd.Series(False, index=df.index)
    return df[col].astype(str).str.strip().isin({">", "<"}) | df[col].astype(str).str.contains(r"[<>]", regex=True, na=False)


def apply_censoring_policy(df: pd.DataFrame, cfg: TargetConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split df into (usable_rows, excluded_rows) according to cfg.policy.
    Returns usable_rows unchanged (policy="clip" does not need to modify
    values here because, by this data contract, the numeric columns
    already hold the boundary value; "clip" just means "don't exclude").
    """
    ic50_censored = _row_is_censored(df, cfg.censor_ic50_col)
    cc50_censored = _row_is_censored(df, cfg.censor_cc50_col)
    any_censored = ic50_censored | cc50_censored

    if cfg.policy == "exclude":
        usable = df[~any_censored].copy()
        excluded = df[any_censored].copy()
    elif cfg.policy == "clip":
        usable = df.copy()
        excluded = df.iloc[0:0].copy()
    else:
        raise ValueError(f"Unknown censor policy: {cfg.policy}")
    return usable, excluded


def compute_log_targets(df: pd.DataFrame, cfg: TargetConfig) -> pd.DataFrame:
    """Return a copy of df with log10_ic50, log10_cc50, log10_si columns added.
    Assumes df has already been through apply_censoring_policy."""
    out = df.copy()
    ic50 = out[cfg.ic50_col].astype(float)
    cc50 = out[cfg.cc50_col].astype(float)
    if (ic50 <= 0).any() or (cc50 <= 0).any():
        raise ValueError(
            "Non-positive IC50/CC50 encountered — cannot take log10. "
            "Check the data contract with A before proceeding."
        )
    out["log10_ic50"] = np.log10(ic50)
    out["log10_cc50"] = np.log10(cc50)
    out["log10_si"] = out["log10_cc50"] - out["log10_ic50"]
    return out
