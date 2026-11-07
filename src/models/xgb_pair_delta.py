"""Train-only paired delta model with the weighted concentration model as fallback."""

from __future__ import annotations

import numpy as np


PAIR_PARAMS = {"n_estimators": 100, "max_depth": 2, "learning_rate": 0.04,
               "min_child_weight": 2, "subsample": 0.85, "colsample_bytree": 0.8,
               "reg_lambda": 5.0, "reg_alpha": 0.1, "random_state": 20261106,
               "n_jobs": 4}


def tanimoto(a, b):
    """Binary Morgan Tanimoto, computed from the first 2048 workflow columns."""
    a, b = np.asarray(a, dtype=bool), np.asarray(b, dtype=bool)
    intersection = np.count_nonzero(a & b, axis=-1)
    union = np.count_nonzero(a | b, axis=-1)
    return np.divide(intersection, union, out=np.zeros_like(intersection, dtype=float), where=union != 0)


def pair_features(query, reference, similarity):
    query, reference = np.asarray(query, dtype=np.float32), np.asarray(reference, dtype=np.float32)
    return np.concatenate((query, reference, query-reference,
                           np.asarray(similarity, dtype=np.float32).reshape(-1, 1)), axis=1)


def build_training_pairs(features, values, weights, max_references, min_similarity):
    """Create directed query/reference examples only from the supplied fit rows."""
    x, values, weights = np.asarray(features), np.asarray(values, float), np.asarray(weights, float)
    valid = np.flatnonzero(np.isfinite(values) & (values > 0))
    pair_x, pair_y, pair_w = [], [], []
    for query in valid:
        candidates = valid[valid != query]
        if not len(candidates):
            continue
        sims = tanimoto(x[candidates, :2048], np.broadcast_to(x[query, :2048], (len(candidates), 2048)))
        eligible = np.flatnonzero(sims >= min_similarity)
        eligible = eligible[np.argsort(sims[eligible])[::-1]][:max_references]
        refs, sim = candidates[eligible], sims[eligible]
        if len(refs):
            pair_x.append(pair_features(np.broadcast_to(x[query], (len(refs), x.shape[1])), x[refs], sim))
            pair_y.append(np.log10(values[query]) - np.log10(values[refs]))
            pair_w.append(np.sqrt(weights[query] * weights[refs]))
    if not pair_x:
        return np.empty((0, x.shape[1]*3+1), dtype=np.float32), np.empty(0), np.empty(0)
    return np.vstack(pair_x), np.concatenate(pair_y), np.concatenate(pair_w)


def assert_reference_partition(train_smiles, held_smiles, scaffold_fn, protocol):
    """Fail closed if a fold's reference bank overlaps held structures/scaffolds."""
    train, held = set(train_smiles), set(held_smiles)
    if train.intersection(held):
        raise AssertionError("held structure leaked into reference bank")
    if protocol == "scaffold":
        train_scaffolds = {scaffold_fn(value) for value in train}
        held_scaffolds = {scaffold_fn(value) for value in held}
        train_scaffolds -= {"acyclic", "invalid"}
        held_scaffolds -= {"acyclic", "invalid"}
        if train_scaffolds.intersection(held_scaffolds):
            raise AssertionError("held scaffold leaked into reference bank")


class XGBPairDelta:
    """Predict each endpoint from query-reference feature differences.

    The reference concentration is adjusted by a learned delta. The median of
    these adjusted estimates is used; raw neighbor labels are never averaged.
    """

    def __init__(self, params=None, pair_params=None, max_references=12, min_similarity=0.35):
        allowed_params = {"n_estimators", "max_depth", "learning_rate", "min_child_weight", "subsample",
                          "colsample_bytree", "reg_lambda", "reg_alpha", "random_state", "n_jobs"}
        unknown = set(params or {}) - allowed_params
        pair_unknown = set(pair_params or {}) - set(PAIR_PARAMS)
        if unknown or pair_unknown:
            raise ValueError(f"unknown model parameters: {sorted(unknown | pair_unknown)}")
        if max_references < 1 or not 0 <= min_similarity <= 1:
            raise ValueError("max_references must be positive and min_similarity in [0,1]")
        self.params = {**(params or {})}
        self.pair_params = {**PAIR_PARAMS, **(pair_params or {})}
        self.max_references = max_references
        self.min_similarity = min_similarity

    def _pair_model(self):
        from xgboost import XGBRegressor
        return XGBRegressor(**self.pair_params, objective="reg:squarederror", tree_method="hist", device="cpu")

    def fit(self, features, ic50, cc50, sample_weight=None):
        x = np.asarray(features, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] < 2048 or not np.isfinite(x).all():
            raise ValueError("features must be finite Morgan2048+descriptor rows")
        targets = {"IC50": np.asarray(ic50, dtype=float), "CC50": np.asarray(cc50, dtype=float)}
        if any(y.shape != (len(x),) for y in targets.values()):
            raise ValueError("endpoint target lengths must match features")
        weights = {name: np.ones(len(x)) if sample_weight is None else np.asarray(sample_weight, dtype=float)
                   for name in targets}
        if any(w.shape != (len(x),) or not np.isfinite(w).all() or (w <= 0).any() for w in weights.values()):
            raise ValueError("sample_weight must contain positive finite row weights")
        from src.models.xgb_weighted import XGBWeightedTwoHead

        # This baseline is fit only on this object's input rows (the caller splits first).
        self.baseline = XGBWeightedTwoHead(self.params).fit(x, targets["IC50"], targets["CC50"], sample_weight=weights["IC50"])
        self.x = x
        self.targets = targets
        self.heads = {}
        self.n_pairs = {}
        for endpoint, values in targets.items():
            xx, yy, ww = build_training_pairs(x, values, weights[endpoint], self.max_references, self.min_similarity)
            if len(yy):
                model = self._pair_model().fit(xx, yy, sample_weight=ww / ww.mean())
                self.heads[endpoint] = model
                self.n_pairs[endpoint] = len(yy)
        return self

    def predict_log_concentrations(self, features):
        q = np.asarray(features, dtype=np.float32)
        result = self.baseline.predict_log_concentrations(q)
        self.last_pair_coverage = np.zeros(len(q), dtype=float)
        for endpoint, model in self.heads.items():
            refs = np.flatnonzero(np.isfinite(self.targets[endpoint]) & (self.targets[endpoint] > 0))
            if not len(refs):
                continue
            adjusted = np.full((len(q), len(refs)), np.nan)
            for i, query in enumerate(q):
                sims = tanimoto(self.x[refs, :2048], np.broadcast_to(query[:2048], (len(refs), 2048)))
                eligible = np.flatnonzero(sims >= self.min_similarity)
                eligible = eligible[np.argsort(sims[eligible])[::-1]][:self.max_references]
                if len(eligible):
                    selected = refs[eligible]
                    features_ = pair_features(np.broadcast_to(query, (len(selected), q.shape[1])), self.x[selected], sims[eligible])
                    adjusted[i, eligible] = np.log10(self.targets[endpoint][selected]) + model.predict(features_)
            covered = np.isfinite(adjusted).any(axis=1)
            result[endpoint][covered] = np.nanmedian(adjusted[covered], axis=1)
            self.last_pair_coverage = np.maximum(self.last_pair_coverage, covered.astype(float))
        return result

    def predict_si(self, features):
        log_conc = self.predict_log_concentrations(features)
        return np.power(10.0, log_conc["CC50"] - log_conc["IC50"])
