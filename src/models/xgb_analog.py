"""Reliability-weighted XGBoost plus a conservative training-analogue predictor."""
import numpy as np
from src.features.fingerprints import MORGAN_BITS
from src.models.xgb_weighted import XGBWeightedTwoHead

NEIGHBOR_SETTINGS = {"k": 5, "min_similarity": 0.5, "similarity_power": 4}


def fingerprint_bits(features):
    matrix = np.asarray(features)
    if matrix.ndim != 2 or matrix.shape[1] < MORGAN_BITS:
        raise ValueError("Analog model requires the fixed Morgan fingerprint feature block")
    bits = matrix[:, :MORGAN_BITS].astype(np.float32)
    if not np.isfinite(bits).all() or not np.isin(bits, [0, 1]).all():
        raise ValueError("Morgan features must be binary")
    return bits


class XGBAnalog(XGBWeightedTwoHead):
    def __init__(self, params=None, clip_si=None):
        super().__init__(params, clip_si)
        self.neighbor_settings = dict(NEIGHBOR_SETTINGS)
        self.blend_weight = 0.0
        self.bank = {}

    def fit(self, features, ic50, cc50, sample_weight=None):
        bits = fingerprint_bits(features)
        super().fit(features, ic50, cc50, sample_weight=sample_weight)
        self.bank = {}
        for name, values in {"IC50": ic50, "CC50": cc50}.items():
            values = np.asarray(values, dtype=float)
            mask = np.isfinite(values) & (values > 0)
            self.bank[name] = {"bits": bits[mask].astype(np.uint8), "log_values": np.log10(values[mask]),
                               "reliability": np.asarray(sample_weight, dtype=float)[mask].copy()}
        return self

    def _neighbor_head(self, bits, name, fallback):
        bank = self.bank[name]
        reference = bank["bits"].astype(np.float32, copy=False)
        # Float accumulation is deliberate: uint8 dot products overflow above 255 bits.
        intersection = bits @ reference.T
        union = bits.sum(axis=1)[:, None] + reference.sum(axis=1)[None, :] - intersection
        similarity = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
        k = min(self.neighbor_settings["k"], len(reference))
        indices = np.argsort(-similarity, axis=1, kind="stable")[:, :k]
        nearest = np.take_along_axis(similarity, indices, axis=1)
        usable = nearest >= self.neighbor_settings["min_similarity"]
        weights = (nearest ** self.neighbor_settings["similarity_power"]
                   * bank["reliability"][indices] * usable)
        totals = weights.sum(axis=1)
        result = np.asarray(fallback, dtype=float).copy()
        available = totals > 0
        result[available] = ((weights * bank["log_values"][indices]).sum(axis=1)[available]
                             / totals[available])
        return result, available, nearest[:, 0], usable.sum(axis=1)

    def predict_components(self, features):
        if set(self.bank) != {"IC50", "CC50"}:
            raise ValueError("Analog model has not been fitted")
        bits = fingerprint_bits(features)
        baseline_logs = super().predict_log_concentrations(features)
        neighbor_logs, available, similarity, counts = {}, {}, {}, {}
        for name in ("IC50", "CC50"):
            neighbor_logs[name], available[name], similarity[name], counts[name] = self._neighbor_head(
                bits, name, baseline_logs[name])
        baseline = 10.0 ** (baseline_logs["CC50"]-baseline_logs["IC50"])
        analog = 10.0 ** (neighbor_logs["CC50"]-neighbor_logs["IC50"])
        covered = available["IC50"] & available["CC50"]
        analog = np.where(covered, analog, baseline)
        if self.clip_si is not None:
            baseline, analog = np.minimum(baseline, self.clip_si), np.minimum(analog, self.clip_si)
        if not np.isfinite(self.blend_weight) or not 0 <= self.blend_weight <= 1:
            raise ValueError("Invalid analogue mixture weight")
        selected = baseline + self.blend_weight * (analog-baseline)
        if not np.isfinite(selected).all() or (selected <= 0).any():
            raise ValueError("Invalid analogue prediction")
        return {"SI_baseline": baseline, "SI_analog": analog, "SI_selected": selected,
            "IC50_base_uM": 10.0 ** baseline_logs["IC50"], "CC50_base_uM": 10.0 ** baseline_logs["CC50"],
            "IC50_analog_uM": 10.0 ** neighbor_logs["IC50"], "CC50_analog_uM": 10.0 ** neighbor_logs["CC50"],
            "analog_available": covered, "IC50_neighbor_similarity": similarity["IC50"],
            "CC50_neighbor_similarity": similarity["CC50"], "IC50_neighbors": counts["IC50"],
            "CC50_neighbors": counts["CC50"]}

    def predict_si(self, features):
        return self.predict_components(features)["SI_selected"]

    def predict_log_si(self, features):
        return np.log10(self.predict_si(features))
