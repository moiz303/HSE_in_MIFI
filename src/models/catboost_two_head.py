"""CatBoost with the same targets, masks and SI calculation as the XGBoost baseline."""

from src.models.xgb_two_head import XGBTwoHead


class CatBoostTwoHead(XGBTwoHead):
    default_params = {
        "iterations": 600,
        "depth": 5,
        "learning_rate": 0.03,
        "l2_leaf_reg": 5.0,
        "random_seed": 20261106,
        "thread_count": 4,
    }

    def _build_head(self):
        from catboost import CatBoostRegressor

        return CatBoostRegressor(
            **self.params, loss_function="RMSE", task_type="CPU",
            verbose=False, allow_writing_files=False,
        )
