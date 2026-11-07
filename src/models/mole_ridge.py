"""Frozen MolE and Morgan feature regressors with separate concentration Ridge heads."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.models.xgb_two_head import exact_target


class RidgeTwoHead:
    def __init__(self, alpha: float = 1.0):
        if not np.isfinite(alpha) or alpha <= 0:
            raise ValueError("Ridge alpha must be positive and finite")
        self.alpha = float(alpha)
        self.heads = {}
        self.n_train = {}

    def fit(self, features, frame):
        features = np.asarray(features, dtype=np.float64)
        if features.ndim != 2 or len(features) != len(frame) or not np.isfinite(features).all():
            raise ValueError("Ridge features must be a finite 2D matrix aligned with targets")
        for name in ("IC50", "CC50"):
            values = exact_target(frame, name)
            mask = np.isfinite(values)
            if not mask.any():
                raise ValueError(f"No exact {name} targets in training fold")
            model = make_pipeline(StandardScaler(), Ridge(alpha=self.alpha))
            model.fit(features[mask], np.log10(values[mask]))
            self.heads[name] = model
            self.n_train[name] = int(mask.sum())
        self.n_features_in_ = features.shape[1]
        return self

    def predict_log_concentrations(self, features):
        features = np.asarray(features, dtype=np.float64)
        if set(self.heads) != {"IC50", "CC50"} or features.ndim != 2 or features.shape[1] != self.n_features_in_:
            raise ValueError("Ridge model is untrained or feature width differs from fit")
        prediction = {name: model.predict(features) for name, model in self.heads.items()}
        if not all(np.isfinite(value).all() for value in prediction.values()):
            raise ValueError("Ridge returned a non-finite concentration prediction")
        return prediction

    def predict_si(self, features):
        log_heads = self.predict_log_concentrations(features)
        return np.power(10.0, np.clip(log_heads["CC50"] - log_heads["IC50"], -300, 300))
