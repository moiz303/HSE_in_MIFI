"""Bounded emphasis on genuine measured high-SI pairs; no inferred concentrations."""
import numpy as np
from src.models.xgb_two_head import exact_target
from src.models.xgb_weighted import XGBWeightedTwoHead

PROFILES = {"weighted": (1, False), "ic2": (2, False),
            "ic4": (4, False), "both2": (2, True)}


def tail_weights(frame, reliability, profile):
    if profile not in PROFILES:
        raise ValueError(f"Unknown tail profile: {profile}")
    base = np.asarray(reliability, dtype=float)
    if base.shape != (len(frame),) or not np.isfinite(base).all() or (base <= 0).any():
        raise ValueError("Positive finite reliability weights required for every row")
    cap, both = PROFILES[profile]
    ic, cc = exact_target(frame, "IC50"), exact_target(frame, "CC50")
    known_measured = frame.target_is_label.astype("string").str.lower().eq("false").fillna(False).to_numpy(bool)
    paired = np.isfinite(ic) & np.isfinite(cc) & known_measured
    factors = np.ones(len(frame))
    # Work in logs to avoid overflowing ratios of extreme finite concentrations.
    factors[paired] = np.exp(np.clip(.5 * (np.log(cc[paired]) - np.log(ic[paired]) - np.log(100)),
                                      0, np.log(cap)))
    return {"IC50": base * factors, "CC50": base * factors if both else base.copy()}


class XGBTail(XGBWeightedTwoHead):
    def __init__(self, params=None, clip_si=None):
        super().__init__(params, clip_si)
        self.profile = "weighted"
