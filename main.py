# Dosya: main.py
import os
import joblib
import nltk
import hashlib
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    balanced_accuracy_score,
    matthews_corrcoef,
    average_precision_score,
    roc_auc_score,
)

# Modüller
from src.data import make_dataset
from src.features import build_features
from src.models import train_model

nltk.download("wordnet", quiet=True)
nltk.download("stopwords", quiet=True)


# -----------------------------
# Helpers
# -----------------------------
def mode_or_nan(s: pd.Series):
    s = s.dropna()
    if len(s) == 0:
        return np.nan
    return s.value_counts().idxmax()


def _sig_from_headlines(headlines: list[str]) -> str:
    """Cache signature: length + first/last headline hash."""
    if len(headlines) == 0:
        raw = "0||"
    else:
        raw = f"{len(headlines)}||{headlines[0]}||{headlines[-1]}"
    return hashlib.md5(raw.encode("utf-8", errors="ignore")).hexdigest()


def compute_or_load_row_finbert(df: pd.DataFrame, cache_dir="data/processed"):
    """
    Satır-bazında (raw rows) FinBERT embedding(768) + sentiment(pos/neg/neu/conf) üretir.
    Cache varsa ve uzunluk+signature uyuyorsa yükler.
    """
    os.makedirs(cache_dir, exist_ok=True)

    emb_path = os.path.join(cache_dir, "finbert_row_emb_768.npy")
    sent_path = os.path.join(cache_dir, "finbert_row_sent.npz")
    sig_path = os.path.join(cache_dir, "finbert_row_cache_sig.txt")

    headlines = df["Headline"].astype(str).tolist()
    sig = _sig_from_headlines(headlines)
    n = len(df)

    if os.path.exists(emb_path) and os.path.exists(sent_path) and os.path.exists(sig_path):
        try:
            with open(sig_path, "r", encoding="utf-8") as f:
                sig_old = f.read().strip()

            emb = np.load(emb_path)
            sent_npz = np.load(sent_path)

            if sig_old == sig and len(emb) == n and len(sent_npz["pos"]) == n:
                print("Loading cached row-level FinBERT (signature+length matched).")
                return emb, sent_npz["pos"], sent_npz["neg"], sent_npz["neu"], sent_npz["conf"]
            else:
                print(
                    "Row-level cache mismatch -> recompute. "
                    f"sig_ok={sig_old == sig}, emb_len={len(emb)}, df_len={n}"
                )
        except Exception as e:
            print(f"Row-level cache load failed -> recompute. Error: {e}")

    print("Computing row-level FinBERT (fresh)...")
    extractor = build_features.FinbertFeatureExtractor()

    emb_list = []
    pos_list, neg_list, neu_list, conf_list = [], [], [], []

    for i, text in enumerate(headlines):
        if i % 200 == 0:
            print(f"  {i}/{n}")
        # embedding (768)
        e = extractor.get_embedding(text)
        emb_list.append(e)

        s = extractor.get_sentiment_scores(text)
        # label casing robust
        s_lower = {str(k).lower(): float(v) for k, v in s.items()} if isinstance(s, dict) else {}
        pos = s_lower.get("positive", 0.0)
        neg = s_lower.get("negative", 0.0)
        neu = s_lower.get("neutral", 0.0)
        conf = max(s_lower.values()) if len(s_lower) else 0.0

        pos_list.append(pos)
        neg_list.append(neg)
        neu_list.append(neu)
        conf_list.append(conf)

    emb = np.array(emb_list, dtype=float)
    pos_arr = np.array(pos_list, dtype=float)
    neg_arr = np.array(neg_list, dtype=float)
    neu_arr = np.array(neu_list, dtype=float)
    conf_arr = np.array(conf_list, dtype=float)

    np.save(emb_path, emb)
    np.savez(sent_path, pos=pos_arr, neg=neg_arr, neu=neu_arr, conf=conf_arr)
    with open(sig_path, "w", encoding="utf-8") as f:
        f.write(sig)

    return emb, pos_arr, neg_arr, neu_arr, conf_arr


def build_daily_panel_with_finbert_agg(df: pd.DataFrame, row_emb, pos, neg, neu, conf) -> pd.DataFrame:
    """
    Market_Index x DateOnly bazında günlük panel:
    - Numeric agregasyon: mean/sum
    - Keyword flags: max
    - Sentiment: mean/max/min/std + conf mean/max
    - Embedding: mean pooling (768)
    """
    rows = []
    g = df.groupby(["Market_Index", "DateOnly"], sort=False)

    for (midx, dateonly), sub in g:
        idx = sub.index.to_numpy()
        if len(idx) == 0:
            continue

        emb_mean = row_emb[idx].mean(axis=0)

        sent_score = (pos[idx] - neg[idx])
        sent_score_mean = float(np.mean(sent_score))
        sent_score_max = float(np.max(sent_score))
        sent_score_min = float(np.min(sent_score))
        sent_score_std = float(np.std(sent_score)) if len(sent_score) > 1 else 0.0

        row = {
            "Market_Index": midx,
            "Date": dateonly,
            "News_Count": int(len(idx)),
            # İsterseniz debug için concat tutabilirsiniz (FinBERT için kullanılmıyor)
            "Headline_Concat": " [SEP] ".join([t for t in sub["Headline"].astype(str).tolist() if t])[:4000],

            "Index_Change_Percent": float(sub["Index_Change_Percent"].mean()) if "Index_Change_Percent" in sub else 0.0,
            "Trading_Volume": float(sub["Trading_Volume"].sum()) if "Trading_Volume" in sub else 0.0,

            "Source": mode_or_nan(sub["Source"]) if "Source" in sub else "UNKNOWN",
            "Market_Event": mode_or_nan(sub["Market_Event"]) if "Market_Event" in sub else "UNKNOWN",
            "Sector": mode_or_nan(sub["Sector"]) if "Sector" in sub else "UNKNOWN",
            "Impact_Level": mode_or_nan(sub["Impact_Level"]) if "Impact_Level" in sub else "UNKNOWN",

            # keyword flags (satır bazında üretildi -> günlük max)
            "keyword_bullish": int(sub["keyword_bullish"].max()) if "keyword_bullish" in sub else 0,
            "keyword_bearish": int(sub["keyword_bearish"].max()) if "keyword_bearish" in sub else 0,
            "mentions_stock": int(sub["mentions_stock"].max()) if "mentions_stock" in sub else 0,

            # sentiment agregasyon
            "sent_pos_mean": float(np.mean(pos[idx])),
            "sent_neg_mean": float(np.mean(neg[idx])),
            "sent_neu_mean": float(np.mean(neu[idx])),
            "sent_conf_mean": float(np.mean(conf[idx])),
            "sent_conf_max": float(np.max(conf[idx])),
            "sent_score_mean": sent_score_mean,
            "sent_score_max": sent_score_max,
            "sent_score_min": sent_score_min,
            "sent_score_std": sent_score_std,

            # embedding (object column)
            "emb_mean": emb_mean,
        }
        rows.append(row)

    panel = pd.DataFrame(rows)
    panel = panel.sort_values(["Market_Index", "Date"]).reset_index(drop=True)
    return panel


def main():
    print("=" * 60)
    print("PROJE PIPELINE: XGBOOST + FINBERT (ROW-AGG) + DAILY PANEL + SVD64 (NEXT-DAY)")
    print("=" * 60)

    # === ADIM 1: Veri Yükleme ===
    print("[1/7] Veri yükleniyor...")
    raw_df = make_dataset.load_raw_data("data/raw/financial_news_market_events_2025.csv")
    df = make_dataset.preprocess_data(raw_df).reset_index(drop=True)

    # Date temizliği
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.tz_localize(None)
    df = df.dropna(subset=["Date", "Market_Index"]).reset_index(drop=True)
    df["DateOnly"] = df["Date"].dt.normalize()

    # Headline temizliği
    df["Headline"] = df["Headline"].fillna("").astype(str)

    # Keyword flags (satır bazında) -> sonra günlük max alacağız
    df = build_features.add_stock_mention_feature(df, headline_col="Headline")
    df = build_features.add_company_mention_feature(df)

    # === ADIM 2: Row-level FinBERT cache/compute ===
    print("[2/7] Row-level FinBERT embedding/sentiment...")
    row_emb, pos, neg, neu, conf = compute_or_load_row_finbert(df)

    # === ADIM 3: Günlük panel + FinBERT agregasyon ===
    print("[3/7] Daily panel + FinBERT aggregation (no truncation)...")
    panel = build_daily_panel_with_finbert_agg(df, row_emb, pos, neg, neu, conf)

    # === ADIM 4: Commodity + Technical (Market_Index bazında) ===
    print("[4/7] Commodity + Technical indicators...")
    panel = build_features.add_commodity_features(panel, date_col="Date")

    # Teknik indikatörleri Market_Index bazında üret (apply warning yerine loop)
    out = []
    for midx, g in panel.groupby("Market_Index", sort=False):
        g = g.sort_values("Date").reset_index(drop=True)
        g = build_features.add_technical_indicators(g, price_col="Index_Change_Percent")
        out.append(g)
    panel = pd.concat(out, ignore_index=True).sort_values(["Market_Index", "Date"]).reset_index(drop=True)

    # Lag (Market_Index bazında)
    panel["Previous_Index_Change_Percent_1d"] = (
        panel.groupby("Market_Index")["Index_Change_Percent"].shift(1).fillna(0.0)
    )

    # === ADIM 5: Target (Next-day) ===
    print("[5/7] Target (next-day) oluşturuluyor...")
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

    # Embedding matrisi (768) — panel filtrelerinden sonra
    emb_matrix = np.vstack(panel["emb_mean"].to_numpy())

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
        "News_Count",

        # sentiment aggregation (multi-headline)
        "sent_pos_mean",
        "sent_neg_mean",
        "sent_neu_mean",
        "sent_conf_mean",
        "sent_conf_max",
        "sent_score_mean",
        "sent_score_max",
        "sent_score_min",
        "sent_score_std",
    ]

    for c in numeric_cols:
        if c not in panel.columns:
            panel[c] = 0.0

    cat_cols = ["Market_Index", "Market_Event", "Sector", "Impact_Level", "Source"]
    for c in cat_cols:
        if c not in panel.columns:
            panel[c] = "UNKNOWN"
    panel[cat_cols] = panel[cat_cols].fillna("UNKNOWN").astype(str)

    X_numeric = panel[numeric_cols].values.astype(float)
    y = panel["Target_Direction"].values.astype(int)

    # === DATE-BASED SPLIT (OUT-OF-TIME) ===
    unique_dates = np.sort(panel["Date"].unique())
    cutoff_date = unique_dates[int(len(unique_dates) * 0.8)]

    train_mask = panel["Date"] <= cutoff_date
    val_mask = panel["Date"] > cutoff_date

    X_num_train = X_numeric[train_mask]
    X_num_val = X_numeric[val_mask]

    X_cat_train = panel.loc[train_mask, cat_cols].values
    X_cat_val = panel.loc[val_mask, cat_cols].values

    X_emb_train = emb_matrix[train_mask.to_numpy()]
    X_emb_val = emb_matrix[val_mask.to_numpy()]

    y_train = y[train_mask]
    y_val = y[val_mask]

    # === ADIM 6: Scaling + OHE + SVD64 ===
    print("[6/7] Scaling + OHE + SVD(64) + concat...")
    scaler = StandardScaler()
    X_num_train_scaled = scaler.fit_transform(X_num_train)
    X_num_val_scaled = scaler.transform(X_num_val)

    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    X_cat_train_ohe = ohe.fit_transform(X_cat_train)
    X_cat_val_ohe = ohe.transform(X_cat_val)

    svd = TruncatedSVD(n_components=64, random_state=42)
    X_emb_train_64 = svd.fit_transform(X_emb_train)
    X_emb_val_64 = svd.transform(X_emb_val)

    X_train_final = np.concatenate([X_num_train_scaled, X_cat_train_ohe, X_emb_train_64], axis=1)
    X_val_final = np.concatenate([X_num_val_scaled, X_cat_val_ohe, X_emb_val_64], axis=1)

    print(f"Eğitim Matrisi Son Boyutu: {X_train_final.shape}")
    print(f"Validation Matrisi Son Boyutu: {X_val_final.shape}")

    # === ADIM 7: Model Eğitimi ===
    print("[7/7] Model eğitiliyor...")
    xgb_model = train_model.train_xgb(
        X_train=X_train_final,
        y_train=y_train,
        X_val=X_val_final,
        y_val=y_val
    )

    # === VALIDATION: Threshold tuning + rapor ===
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
    joblib.dump(svd, "models/svd.pkl")

    print("\nBAŞARILI! Model kaydedildi.")
    print("models/xgb_model.pkl, models/scaler.pkl, models/ohe.pkl, models/svd.pkl, models/threshold.pkl")


if __name__ == "__main__":
    main()
