# from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import xgboost as xgb

from xgboost.callback import EarlyStopping

import numpy as np
from xgboost import XGBClassifier

class XGBBoosterWrapper:
    """
    xgboost.train() Booster'ını sklearn benzeri arayüzle kullanmak için.
    main.py'nin predict_proba çağrıları bozulmasın diye wrapper.
    """
    def __init__(self, booster: xgb.Booster):
        self.booster = booster

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        d = xgb.DMatrix(X)
        p = self.booster.predict(d)  # binary:logistic -> P(Up)
        p = np.clip(p, 1e-7, 1 - 1e-7)
        return np.vstack([1 - p, p]).T

    def predict(self, X: np.ndarray) -> np.ndarray:
        # sklearn gibi 0.5 threshold
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def train_xgb(X_train, y_train, X_val, y_val):
    # Up sınıfını biraz daha maliyetli yap
    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval   = xgb.DMatrix(X_val, label=y_val)

    params = {
        "objective": "binary:logistic",
        "tree_method": "hist",
        "seed": 42,
        "eta": 0.03,
        "max_depth": 4,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.6,
        "lambda": 2.0,
        "alpha": 0.0,
        "scale_pos_weight": float(scale_pos_weight),
        "eval_metric": ["auc", "aucpr", "logloss"],
    }

    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=4000,
        evals=[(dval, "validation")],
        early_stopping_rounds=80,
        verbose_eval=200
    )

    return XGBBoosterWrapper(booster)
