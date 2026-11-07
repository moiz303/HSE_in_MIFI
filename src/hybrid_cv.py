"""Choose a conservative mixture weight inside train; fixed validation stays external."""
import numpy as np
from src import splits
from src.data.kaggle_reference import original_pairs
from src.models.xgb_two_head import exact_target
from src.models.xgb_hybrid import XGBHybrid

WEIGHTS = (0.0, 0.1, 0.25, 0.5, 1.0)


def classifier_labels(frame):
    source = frame.source.fillna("").str.split(";").apply(lambda v: "kaggle_train" in v).to_numpy()
    flags = frame.target_is_label.astype("string").fillna("unknown").str.lower()
    labels = np.full(len(frame), np.nan)
    known = source & flags.isin(["true", "false"]).to_numpy()
    labels[known] = flags.eq("true").to_numpy(dtype=bool)[known]
    return labels


def choose_weight(scores):
    """Require >=2% lower mean fold MSE and wins in at least two folds."""
    errors = np.asarray(scores, dtype=float)  # folds x candidates, mean squared errors
    baseline = errors[:, 0]
    eligible = [i for i in range(1, len(WEIGHTS))
                if errors[:, i].mean() <= .98 * baseline.mean()
                and int((errors[:, i] < baseline).sum()) >= 2]
    winner = min(eligible, key=lambda i: (errors[:, i].mean(), WEIGHTS[i])) if eligible else 0
    return WEIGHTS[winner]


def select_on_train(frame, matrix, reference, params):
    smiles = frame.standard_smiles.tolist()
    groups = splits.scaffold_split(smiles, fractions={str(i): 1/3 for i in range(3)}, seed=20261106)
    splits.assert_no_leaks(smiles, groups)
    scores, details = [], []
    for name, indices in groups.items():
        test = np.asarray(indices, dtype=int)
        train = np.setdiff1d(np.arange(len(frame)), test)
        if not len(test) or not len(train):
            raise ValueError("Not enough scaffold groups for three-fold CV")
        model = XGBHybrid(params)
        print(f"[CV {name}/3] train={len(train)}, validation={len(test)}", flush=True)
        model.fit(matrix[train], exact_target(frame.iloc[train], "IC50"), exact_target(frame.iloc[train], "CC50"))
        model.fit_classifier(matrix[train], classifier_labels(frame.iloc[train]))
        baseline = model.predict_baseline_si(matrix[test])
        candidate = model.predict_candidate_si(matrix[test])
        row = []
        for weight in WEIGHTS:
            truth, prediction = original_pairs(frame.iloc[test], baseline + weight*(candidate-baseline), reference)
            if not len(truth):
                raise ValueError("CV fold contains no original competition targets")
            row.append(float(np.mean((truth-prediction)**2)))
        scores.append(row)
        details.append({"fold": name, "n_competition": len(truth), "mse": row})
        print(f"[CV {name}/3] weights={WEIGHTS}; MSE={row}", flush=True)
    weight = choose_weight(scores)
    return {"weight": weight, "weights": list(WEIGHTS), "folds": details,
            "rule": ">=2% lower mean fold MSE and wins in >=2/3 train folds"}
