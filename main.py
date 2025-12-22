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
    if len(headlines) == 0:
        raw = "0||"
    else:
        raw = f"{len(headlines)}||{headlines[0]}||{headlines[-1]}"
    return hashlib.md5(raw.encode("utf-8", errors="ignore")).hexdigest()


def compute_or_load_row_finbert(df: pd.DataFrame, cache_dir="data/processed"):
    """
    Satır-bazında FinBERT embedding(768) + sentiment(pos/neg/neu/conf) üretir.
    Cache varsa ve signature+length uyuyorsa yükler.
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

        e = extractor.get_embedding(text)  # (768,)
        emb_list.append(e)

        s = extractor.get_sentiment_scores(text)
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
    Market_Index x DateOnly günlük panel:
      - Numeric: mean/sum
      - Keyword: max
      - Sentiment: mean/max/min/std
      - Embedding: mean pooling + (opsiyonel) std_mean
    """
    rows = []
    g = df.groupby(["Market_Index", "DateOnly"], sort=False)

    for (midx, dateonly), sub in g:
        idx = sub.index.to_numpy()
        if len(idx) == 0:
            continue

        emb_block = row_emb[idx]
        emb_mean = emb_block.mean(axis=0)
        emb_std_mean = float(emb_block.std(axis=0).mean()) if len(idx) > 1 else 0.0

        sent_score = (pos[idx] - neg[idx])
        sent_score_mean = float(np.mean(sent_score))
        sent_score_max = float(np.max(sent_score))
        sent_score_min = float(np.min(sent_score))
        sent_score_std = float(np.std(sent_score)) if len(sent_score) > 1 else 0.0

        row = {
            "Market_Index": midx,
            "Date": dateonly,
            "News_Count": int(len(idx)),
            "Headline_Concat": " [SEP] ".join([t for t in sub["Headline"].astype(str).tolist() if t])[:4000],

            "Index_Change_Percent": float(sub["Index_Change_Percent"].mean()) if "Index_Change_Percent" in sub else 0.0,
            "Trading_Volume": float(sub["Trading_Volume"].sum()) if "Trading_Volume" in sub else 0.0,

            "Source": mode_or_nan(sub["Source"]) if "Source" in sub else "UNKNOWN",
            "Market_Event": mode_or_nan(sub["Market_Event"]) if "Market_Event" in sub else "UNKNOWN",
            "Sector": mode_or_nan(sub["Sector"]) if "Sector" in sub else "UNKNOWN",
            "Impact_Level": mode_or_nan(sub["Impact_Level"]) if "Impact_Level" in sub else "UNKNOWN",

            "keyword_bullish": int(sub["keyword_bullish"].max()) if "keyword_bullish" in sub else 0,
            "keyword_bearish": int(sub["keyword_bearish"].max()) if "keyword_bearish" in sub else 0,
            "mentions_stock": int(sub["mentions_stock"].max()) if "mentions_stock" in sub else 0,

            "sent_pos_mean": float(np.mean(pos[idx])),
            "sent_neg_mean": float(np.mean(neg[idx])),
            "sent_neu_mean": float(np.mean(neu[idx])),
            "sent_conf_mean": float(np.mean(conf[idx])),
            "sent_conf_max": float(np.max(conf[idx])),

            "sent_score_mean": sent_score_mean,
            "sent_score_max": sent_score_max,
            "sent_score_min": sent_score_min,
            "sent_score_std": sent_score_std,

            "emb_mean": emb_mean,
            "emb_std_mean": emb_std_mean,
        }
        rows.append(row)

    panel = pd.DataFrame(rows).sort_values(["Market_Index", "Date"]).reset_index(drop=True)
    return panel


def add_return_dynamics(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Next-day için en kritik sinyaller: ret_t0, multi-lag, rolling mean/std.
    Tüm rolling'ler shift(1) ile yapılır (tamamen geçmişe dayalı).
    """
    panel = panel.sort_values(["Market_Index", "Date"]).reset_index(drop=True)

    # takvim
    panel["dow"] = panel["Date"].dt.dayofweek.astype(int)
    panel["month"] = panel["Date"].dt.month.astype(int)

    # bugünün return'ü (t0)
    panel["ret_t0"] = panel["Index_Change_Percent"].astype(float)

    # lag returns
    for lag in [1, 2, 3, 5]:
        panel[f"lag_ret_{lag}"] = (
            panel.groupby("Market_Index")["Index_Change_Percent"].shift(lag).fillna(0.0)
        )

    # rolling stats (shift(1) ile)
    for w in [3, 5, 10, 20]:
        panel[f"roll_mean_{w}"] = panel.groupby("Market_Index")["Index_Change_Percent"].transform(
            lambda s: s.shift(1).rolling(window=w).mean()
        ).fillna(0.0)
        panel[f"roll_std_{w}"] = panel.groupby("Market_Index")["Index_Change_Percent"].transform(
            lambda s: s.shift(1).rolling(window=w).std()
        ).fillna(0.0)

    return panel


def main():
    print("=" * 60)
    print("PROJE PIPELINE: XGBOOST + FINBERT (ROW-AGG) + DAILY PANEL + SVD64 + RETURN-DYNAMICS (NEXT-DAY)")
    print("=" * 60)

    # === ADIM 1: Veri Yükleme ===
    print("[1/7] Veri yükleniyor...")
    raw_df = make_dataset.load_raw_data("data/raw/financial_news_market_events_2025.csv")
    df = make_dataset.preprocess_data(raw_df).reset_index(drop=True)

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.tz_localize(None)
    df = df.dropna(subset=["Date", "Market_Index"]).reset_index(drop=True)
    df["DateOnly"] = df["Date"].dt.normalize()
    df["Headline"] = df["Headline"].fillna("").astype(str)

    # satır bazında keyword
    df = build_features.add_stock_mention_feature(df, headline_col="Headline")
    df = build_features.add_company_mention_feature(df)

    # === ADIM 2: Row-level FinBERT ===
    print("[2/7] Row-level FinBERT embedding/sentiment...")
    row_emb, pos, neg, neu, conf = compute_or_load_row_finbert(df)

    # === ADIM 3: Daily panel + agregasyon ===
    print("[3/7] Daily panel + FinBERT aggregation...")
    panel = build_daily_panel_with_finbert_agg(df, row_emb, pos, neg, neu, conf)

    # === ADIM 4: Commodity + Technical + Return dynamics ===
    print("[4/7] Commodity + Technical indicators + return dynamics...")
    panel = build_features.add_commodity_features(panel, date_col="Date")

    out = []
    for midx, g in panel.groupby("Market_Index", sort=False):
        g = g.sort_values("Date").reset_index(drop=True)
        g = build_features.add_technical_indicators(g, price_col="Index_Change_Percent")
        out.append(g)
    panel = pd.concat(out, ignore_index=True).sort_values(["Market_Index", "Date"]).reset_index(drop=True)

    # lag1 (mevcut projede kullanılan isim)
    panel["Previous_Index_Change_Percent_1d"] = (
        panel.groupby("Market_Index")["Index_Change_Percent"].shift(1).fillna(0.0)
    )

    # ek dinamik feature'lar
    panel = add_return_dynamics(panel)

    # === ADIM 5: Target (Next-day) ===
    print("[5/7] Target (next-day) oluşturuluyor...")
    panel["return_1d"] = panel.groupby("Market_Index")["Index_Change_Percent"].shift(-1)
    panel = panel.dropna(subset=["return_1d"]).reset_index(drop=True)

    # --- Adaptive (volatility-scaled) neutral-drop labeling ---
    # roll_std_10 zaten add_return_dynamics() içinde shift(1) ile üretildiği için leakage yapmaz.
    K_VOL = 0.90  # 0.70 / 0.90 / 1.10 gibi değerlerle deneyebilirsin
    MIN_THR = 0.15  # taban eşik: çok küçük threshold'ları engeller

    if "roll_std_10" not in panel.columns:
        # Güvenlik: eğer kolon yoksa üret (tamamen geçmişe dayalı)
        panel["roll_std_10"] = panel.groupby("Market_Index")["Index_Change_Percent"].transform(
            lambda s: s.shift(1).rolling(window=10).std()
        ).fillna(0.0)

    thr_vec = np.maximum(MIN_THR, K_VOL * panel["roll_std_10"].astype(float))
    r = panel["return_1d"].astype(float)

    panel["Target_Direction"] = np.where(
        r > thr_vec, 1,
        np.where(r < -thr_vec, 0, np.nan)
    )

    before = len(panel)
    panel = panel.dropna(subset=["Target_Direction"]).reset_index(drop=True)
    panel["Target_Direction"] = panel["Target_Direction"].astype(int)

    print(f"Adaptive labeling: kept {len(panel)}/{before} rows | K_VOL={K_VOL} MIN_THR={MIN_THR}")

    print(f"Eğitime girecek günlük satır sayısı: {len(panel)}")
    print(panel["Target_Direction"].value_counts(normalize=True))

    # embedding matrisi
    emb_matrix = np.vstack(panel["emb_mean"].to_numpy())

    # === FEATURE SET ===
    numeric_cols = [
        # base numeric
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

        # sentiment aggregation
        "sent_pos_mean",
        "sent_neg_mean",
        "sent_neu_mean",
        "sent_conf_mean",
        "sent_conf_max",
        "sent_score_mean",
        "sent_score_max",
        "sent_score_min",
        "sent_score_std",

        # embedding disagreement
        "emb_std_mean",

        # return dynamics
        "ret_t0",
        "lag_ret_1",
        "lag_ret_2",
        "lag_ret_3",
        "lag_ret_5",
        "roll_mean_3",
        "roll_mean_5",
        "roll_mean_10",
        "roll_mean_20",
        "roll_std_3",
        "roll_std_5",
        "roll_std_10",
        "roll_std_20",

        # calendar
        "dow",
        "month",
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

    # === ADIM 6: Scaling + OHE + SVD ===
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

    SVD_DIM = 64
    svd = TruncatedSVD(n_components=SVD_DIM, random_state=42)
    X_emb_train_k = svd.fit_transform(X_emb_train)
    X_emb_val_k = svd.transform(X_emb_val)

    X_train_final = np.concatenate([X_num_train_scaled, X_cat_train_ohe, X_emb_train_k], axis=1)
    X_val_final = np.concatenate([X_num_val_scaled, X_cat_val_ohe, X_emb_val_k], axis=1)

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

    # === VALIDATION ===
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

    # === SAVE ===
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
