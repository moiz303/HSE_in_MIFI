"""Concentration baseline plus an experimental classifier of the SI=10000 label."""
import numpy as np
from xgboost import XGBClassifier
from src.models.xgb_two_head import XGBTwoHead


CLASSIFIER_PARAMS = {
    "n_estimators": 250, "max_depth": 2, "learning_rate": 0.03,
    "min_child_weight": 3, "reg_lambda": 10.0,
    "subsample": 0.8, "colsample_bytree": 0.8,
    "random_state": 20261106, "n_jobs": 4,
}


def select_hybrid(baseline_mse, candidate_mse):
    """Predeclared validation rule; not a statistical significance claim."""
    return (baseline_mse is not None and candidate_mse is not None
            and np.isfinite(baseline_mse) and np.isfinite(candidate_mse)
            and baseline_mse > 0 and candidate_mse <= 0.98 * baseline_mse)


class XGBHybrid(XGBTwoHead):
    def __init__(self, params=None, clip_si=None):
        super().__init__(params, clip_si)
        self.classifier = None
        self.single_class = None
        self.use_classifier = False

    def fit_classifier(self, features, labels):
        labels = np.asarray(labels, dtype=float)
        mask = np.isfinite(labels)
        if labels.shape != (len(features),) or not mask.any():
            raise ValueError("classifier requires aligned, observed training labels")
        if not np.isin(labels[mask], [0, 1]).all():
            raise ValueError("classifier labels must be 0 or 1")
        self.use_classifier = False
        self.classifier = None
        self.single_class = None
        self.n_train["classifier"] = int(mask.sum())
        self.n_train["classifier_positive"] = int(labels[mask].sum())
        print(f"[fit] label classifier: {self.n_train['classifier']} rows, "
              f"{self.n_train['classifier_positive']} positive; params={CLASSIFIER_PARAMS}", flush=True)
        unique = np.unique(labels[mask])
        if len(unique) == 1:
            self.single_class = float(unique[0])
        else:
            self.classifier = XGBClassifier(**CLASSIFIER_PARAMS, objective="binary:logistic",
                                            eval_metric="logloss", tree_method="hist", device="cpu")
            self.classifier.fit(features[mask], labels[mask])
        return self

    def predict_label_probability(self, features):
        if self.single_class is not None:
            return np.full(len(features), self.single_class)
        if self.classifier is None:
            raise ValueError("label classifier has not been fitted")
        return self.classifier.predict_proba(features)[:, 1]

    def predict_baseline_si(self, features):
        return 10.0 ** super().predict_log_si(features)

    def predict_candidate_si(self, features):
        probability = self.predict_label_probability(features)
        baseline = self.predict_baseline_si(features)
        return probability * 10000.0 + (1.0 - probability) * baseline

    def predict_si(self, features):
        return (self.predict_candidate_si(features) if self.use_classifier
                else self.predict_baseline_si(features))

    def predict_log_si(self, features):
        return np.log10(self.predict_si(features))

    def __getstate__(self):
        state = super().__getstate__()
        if self.classifier is not None:
            state["_native_classifier"] = self.classifier.get_booster().save_raw(raw_format="ubj")
            state["classifier"] = None
        return state

    def __setstate__(self, state):
        native = state.pop("_native_classifier", None)
        super().__setstate__(state)
        if native is not None:
            self.classifier = XGBClassifier(**CLASSIFIER_PARAMS)
            self.classifier.load_model(native)
