"""Five independent concentration pairs; arithmetic mean of their SI predictions."""
import numpy as np
from src.models.xgb_two_head import XGBTwoHead


class XGBEnsemble(XGBTwoHead):
    def __init__(self, params=None, clip_si=None):
        super().__init__(params, clip_si)
        self.seeds = [self.params["random_state"] + offset for offset in range(5)]
        self.members = []

    def fit(self, features, ic50, cc50):
        self.members = []
        for index, seed in enumerate(self.seeds, 1):
            print(f"[ensemble {index}/{len(self.seeds)}] seed={seed} starting", flush=True)
            member = XGBTwoHead({**self.params, "random_state": seed}, self.clip_si)
            member.fit(features, ic50, cc50)
            self.members.append(member)
            print(f"[ensemble {index}/{len(self.seeds)}] finished", flush=True)
        self.n_features_in_ = self.members[0].n_features_in_
        self.n_train = self.members[0].n_train.copy()
        return self

    def _check_fitted(self):
        if len(self.members) != len(self.seeds):
            raise ValueError("All five ensemble members must be fitted")

    def predict_members_si(self, features):
        self._check_fitted()
        return np.column_stack([member.predict_si(features) for member in self.members])

    def predict_si(self, features):
        return self.predict_members_si(features).mean(axis=1)

    def predict_log_si(self, features):
        return np.log10(self.predict_si(features))

    def predict_log_concentrations(self, features):
        self._check_fitted()
        logs = [member.predict_log_concentrations(features) for member in self.members]
        # Concentration diagnostics: mean concentration, then log. SI is averaged separately.
        return {name: np.log10(np.mean([10.0 ** log[name] for log in logs], axis=0))
                for name in ("IC50", "CC50")}
