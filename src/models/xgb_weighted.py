"""Fixed reliability-weighted concentration baseline, no classifier or weight search."""
from src.models.xgb_two_head import XGBTwoHead


class XGBWeightedTwoHead(XGBTwoHead):
    def fit(self, features, ic50, cc50, sample_weight=None):
        if sample_weight is None:
            raise ValueError("Reliability-weighted model requires explicit sample weights")
        return super().fit(features, ic50, cc50, sample_weight=sample_weight)
