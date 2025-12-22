# src/models/train_model.py
import os
import json
import numpy as np

from xgboost import XGBClassifier


def _get_env_int(name: str, default: int) -> int:
    v = os.getenv(name, str(default))
    try:
        return int(float(v))
    except Exception:
        return default


def _get_env_float(name: str, default: float) -> float:
    v = os.getenv(name, str(default))
    try:
        return float(v)
    except Exception:
        return default


def _get_env_str(name: str, default: str) -> str:
    v = os.getenv(name, default)
    return str(v) if v is not None else default


def get_xgb_params_from_env() -> dict:
    """
    XGB hiperparametrelerini ENV üzerinden alır.
    Varsayılanlar: küçük veri + overfit azaltma odaklı.
    """
    params = {
        "n_estimators": _get_env_int("XGB_N_ESTIMATORS", 4000),
        "learning_rate": _get_env_float("XGB_LEARNING_RATE", 0.03),
        "max_depth": _get_env_int("XGB_MAX_DEPTH", 3),
        "min_child_weight": _get_env_float("XGB_MIN_CHILD_WEIGHT", 10.0),
        "subsample": _get_env_float("XGB_SUBSAMPLE", 0.8),
        "colsample_bytree": _get_env_float("XGB_COLSAMPLE_BYTREE", 0.7),
        "gamma": _get_env_float("XGB_GAMMA", 0.0),
        "reg_lambda": _get_env_float("XGB_REG_LAMBDA", 10.0),
        "reg_alpha": _get_env_float("XGB_REG_ALPHA", 0.0),

        # Sınıflar neredeyse dengeli; istersen aç:
        # "scale_pos_weight": _get_env_float("XGB_SCALE_POS_WEIGHT", 1.0),

        "objective": "binary:logistic",
        "random_state": _get_env_int("XGB_RANDOM_STATE", 42),
        "n_jobs": _get_env_int("XGB_N_JOBS", -1),

        # XGBoost sklearn API: fit'e değil, constructor'a verilir. :contentReference[oaicite:1]{index=1}
        "eval_metric": ["auc", "aucpr", "logloss"],
        "early_stopping_rounds": _get_env_int("XGB_EARLY_STOPPING", 150),

        # CPU için güvenli default:
        "tree_method": _get_env_str("XGB_TREE_METHOD", "hist"),
    }

    # İstersen JSON ile komple override:
    # $env:XGB_PARAMS_JSON='{"max_depth":4,"reg_lambda":5.0}'
    raw = os.getenv("XGB_PARAMS_JSON", "").strip()
    if raw:
        try:
            override = json.loads(raw)
            if isinstance(override, dict):
                params.update(override)
        except Exception:
            pass

    return params


def train_xgb(X_train, y_train, X_val, y_val, verbose: int = 200):
    """
    XGBClassifier eğitir ve döndürür.
    """
    params = get_xgb_params_from_env()

    model = XGBClassifier(**params)

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=verbose
    )

    return model
