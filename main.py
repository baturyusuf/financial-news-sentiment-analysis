# Dosya: main.py
import os
import joblib
import nltk
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    balanced_accuracy_score,
    matthews_corrcoef,
    average_precision_score,
    roc_auc_score
)

# Modüller
from src.data import make_dataset
from src.features import build_features
from src.models import train_model

nltk.download("wordnet", quiet=True)
nltk.download("stopwords", quiet=True)


def mode_or_nan(s: pd.Series):
    s = s.dropna()
    if len(s) == 0:
        return np.nan
    return s.value_counts().idxmax()


def main():
    print("=" * 60)
    print("PROJE PIPELINE: XGBOOST + FINBERT + DAILY PANEL (NEXT-DAY)")
    print("=" * 60)

    # === ADIM 1: Veri Yükleme ===
    print("[1/6] Veri yükleniyor...")
    raw_df = make_dataset.load_raw_data("data/raw/financial_news_market_events_2025.csv")
    processed_df = make_dataset.preprocess_data(raw_df).reset_index(drop=True)

    # === ADIM 2: DAILY PANEL OLUŞTURMA (Market_Index x Date) ===
    print("[2/6] Daily panel hazırlanıyor (Market_Index x Date)...")
    df = processed_df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.tz_localize(None)
    df = df.dropna(subset=["Date", "Market_Index"]).reset_index(drop=True)

    df["DateOnly"] = df["Date"].dt.normalize()
    df["Headline"] = df["Headline"].fillna("").astype(str)

    panel = (
        df.groupby(["Market_Index", "DateOnly"], as_index=False)
          .agg(
              Date=("DateOnly", "first"),
              News_Count=("Headline", "size"),
              Headline=("Headline", lambda x: " [SEP] ".join([t for t in x.tolist() if t])),
              Index_Change_Percent=("Index_Change_Percent", "mean"),
              Trading_Volume=("Trading_Volume", "sum"),
              Source=("Source", mode_or_nan),
              Market_Event=("Market_Event", mode_or_nan),
              Sector=("Sector", mode_or_nan),
              Impact_Level=("Impact_Level", mode_or_nan),
          )
    )

    panel = panel.sort_values(["Market_Index", "Date"]).reset_index(drop=True)

    # Emtia feature'ları (date bazlı merge_asof)
    panel = build_features.add_commodity_features(panel, date_col="Date")

    # Teknik indikatörler (NOT: bu haliyle leakage içerir; next-day için sorun değil, ama yine de groupby şart)
    panel = panel.groupby("Market_Index", group_keys=False).apply(
        lambda g: build_features.add_technical_indicators(g, price_col="Index_Change_Percent")
    )

    # Lag (endeks bazında)
    panel["Previous_Index_Change_Percent_1d"] = (
        panel.groupby("Market_Index")["Index_Change_Percent"].shift(1).fillna(0)
    )

    # Keyword features (Headline günlük birleştirilmiş metin)
    panel = build_features.add_stock_mention_feature(panel, headline_col="Headline")
    panel = build_features.add_company_mention_feature(panel)

    # =========================
    # TARGET (NEXT-DAY) OLUŞTURMA (Market_Index bazında)
    # =========================
    panel["return_1d"] = panel.groupby("Market_Index")["Index_Change_Percent"].shift(-1)
    panel = panel.dropna(subset=["return_1d"]).reset_index(drop=True)

    THRESHOLD = 0.2
    r = panel["return_1d"]
    panel["Target_Direction"] = np.where(
        r > THRESHOLD, 1,
        np.where(r < -THRESHOLD, 0, np.nan)
    )
    panel = panel.dropna(subset=["Target_Direction"]).reset_index(drop=True)
    panel["Target_Direction"] = panel["Target_Direction"].astype(int)

    print(f"Eğitime girecek günlük satır sayısı: {len(panel)}")
    print(panel["Target_Direction"].value_counts(normalize=True))

    # === ADIM 3: FinBERT Embedding + Sentiment (GÜNLÜK) ===
    print("[3/6] Günlük embedding/sentiment hesaplanıyor...")
    extractor = build_features.FinbertFeatureExtractor()

    headlines = panel["Headline"].tolist()
    embeddings = []
    sentiments = []

    total = len(headlines)
    for i, text in enumerate(headlines):
        if i % 200 == 0:
            print(f"  Embedding: {i}/{total}")
        embeddings.append(extractor.get_embedding(text))
        sentiments.append(extractor.get_sentiment_scores(text))

    embeddings = np.array(embeddings)

    panel["sent_score"] = [s.get("positive", 0.0) - s.get("negative", 0.0) for s in sentiments]
    panel["sent_conf"] = [max(s.values()) if isinstance(s, dict) and len(s) else 0.0 for s in sentiments]

    # === FEATURE SET ===
    numeric_cols = [
        "Trading_Volume",
        "Previous_Index_Change_Percent_1d",
        "Volatility_7d",
        "RSI_14",
        "Cumulative_Return_3d",
        "Gold_close_t1",
        "Gold_ret1_t1",
        "Oil_close_t1",
        "Oil_ret1_t1",
        "keyword_bullish",
        "keyword_bearish",
        "mentions_stock",
        "sent_score",
        "sent_conf",
        "News_Count",
    ]

    for c in numeric_cols:
        if c not in panel.columns:
            panel[c] = 0

    cat_cols = ["Market_Index", "Market_Event", "Sector", "Impact_Level", "Source"]
    for c in cat_cols:
        if c not in panel.columns:
            panel[c] = "UNKNOWN"
    panel[cat_cols] = panel[cat_cols].fillna("UNKNOWN").astype(str)

    X_numeric = panel[numeric_cols].values
    y = panel["Target_Direction"].values

    # === DATE-BASED SPLIT (OUT-OF-TIME) ===
    unique_dates = np.sort(panel["Date"].unique())
    cutoff_date = unique_dates[int(len(unique_dates) * 0.8)]

    train_mask = panel["Date"] <= cutoff_date
    val_mask = panel["Date"] > cutoff_date

    X_num_train = X_numeric[train_mask]
    X_num_val = X_numeric[val_mask]
    X_emb_train = embeddings[train_mask]
    X_emb_val = embeddings[val_mask]
    y_train = y[train_mask]
    y_val = y[val_mask]

    X_cat_train = panel.loc[train_mask, cat_cols].values
    X_cat_val = panel.loc[val_mask, cat_cols].values

    # === ADIM 5: Scaling + OHE + Birleştirme ===
    print("[5/6] Scaling + OHE + Embedding Birleştirme...")
    scaler = StandardScaler()
    X_num_train_scaled = scaler.fit_transform(X_num_train)
    X_num_val_scaled = scaler.transform(X_num_val)

    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    X_cat_train_ohe = ohe.fit_transform(X_cat_train)
    X_cat_val_ohe = ohe.transform(X_cat_val)

    X_train_final = np.concatenate([X_num_train_scaled, X_cat_train_ohe, X_emb_train], axis=1)
    X_val_final = np.concatenate([X_num_val_scaled, X_cat_val_ohe, X_emb_val], axis=1)

    print(f"Eğitim Matrisi Son Boyutu: {X_train_final.shape}")
    print(f"Validation Matrisi Son Boyutu: {X_val_final.shape}")

    # === ADIM 6: Model Eğitimi ===
    print("[6/6] Model Eğitiliyor...")
    xgb_model = train_model.train_xgb(
        X_train=X_train_final,
        y_train=y_train,
        X_val=X_val_final,
        y_val=y_val
    )

    # =========================
    # VALIDATION: Threshold Tuning + Raporlama
    # =========================
    val_probs = xgb_model.predict_proba(X_val_final)[:, 1]

    pr_auc = average_precision_score(y_val, val_probs)
    roc_auc = roc_auc_score(y_val, val_probs)
    print(f"[Validation] PR-AUC (Average Precision): {pr_auc:.4f}")
    print(f"[Validation] ROC-AUC                : {roc_auc:.4f}")

    best = None
    for thr in np.linspace(0.05, 0.95, 91):
        val_pred = (val_probs >= thr).astype(int)
        bacc = balanced_accuracy_score(y_val, val_pred)
        mcc = matthews_corrcoef(y_val, val_pred)
        score = bacc
        if best is None or score > best["score"]:
            best = {"thr": float(thr), "score": float(score), "bacc": float(bacc), "mcc": float(mcc)}

    print(f"[Threshold Search] Best threshold: {best['thr']:.2f} | BalancedAcc={best['bacc']:.4f} | MCC={best['mcc']:.4f}")

    val_pred_best = (val_probs >= best["thr"]).astype(int)
    print("\n[Validation] Confusion Matrix (best thr)")
    print(confusion_matrix(y_val, val_pred_best))
    print("\n[Validation] Classification Report (best thr)")
    print(classification_report(y_val, val_pred_best, target_names=["Down", "Up"]))

    # === MODELLERİ KAYDET ===
    os.makedirs("models", exist_ok=True)
    joblib.dump(best["thr"], "models/threshold.pkl")
    joblib.dump(xgb_model, "models/xgb_model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(numeric_cols, "models/numeric_cols.pkl")
    joblib.dump(cat_cols, "models/cat_cols.pkl")
    joblib.dump(ohe, "models/ohe.pkl")

    print("\nBAŞARILI! Model kaydedildi.")
    print("models/xgb_model.pkl, models/scaler.pkl, models/ohe.pkl, models/threshold.pkl")


if __name__ == "__main__":
    main()
