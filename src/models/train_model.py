from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score


def train_xgb(
    X_train,
    y_train,
    X_val,
    y_val
):
    model = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        tree_method="hist"  # CPU için hızlı
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]

    acc = accuracy_score(y_val, y_pred)
    auc = roc_auc_score(y_val, y_prob)

    print(f"[XGBoost] Validation Accuracy: {acc:.4f}")
    print(f"[XGBoost] Validation AUC     : {auc:.4f}")

    return model
