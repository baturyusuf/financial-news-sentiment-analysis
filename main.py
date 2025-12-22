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


def search_best_threshold(y_true: np.ndarray, prob: np.ndarray, objective: str = "mcc"):
    """
    objective: "mcc" or "bacc"
    Returns dict: {"thr","score","bacc","mcc"}
    """
    objective = (objective or "mcc").lower().strip()
    best = None

    for thr in np.linspace(0.05, 0.95, 91):
        pred = (prob >= thr).astype(int)
        bacc = float(balanced_accuracy_score(y_true, pred))
        mcc = float(matthews_corrcoef(y_true, pred))

        if objective in ("mcc", "matthews"):
            score = mcc
        else:
            score = bacc

        if best is None or score > best["score"]:
            best = {"thr": float(thr), "score": float(score), "bacc": bacc, "mcc": mcc}

    return best


def _sig_from_headlines(headlines: list[str]) -> str:
    """
    Cache signature: length + first/last 25 headlines.
    (Hızlı ve pratik; tüm dataset'i hashlemiyoruz.)
    """
    n = len(headlines)
    head = headlines[:25]
    tail = headlines[-25:] if n >= 25 else headlines
    raw = "||".join([str(n)] + head + ["<MID>"] + tail)
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
      - Embedding: mean pooling + disagreement (std mean)
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
            "Date": pd.to_datetime(dateonly).tz_localize(None),
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

            "emb_mean": emb_mean,          # ndarray(768,)
            "emb_std_mean": emb_std_mean,  # float
        }
        rows.append(row)

    panel = pd.DataFrame(rows).sort_values(["Market_Index", "Date"]).reset_index(drop=True)
    return panel


def add_return_dynamics(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Next-day için kritik sinyaller: ret_t0, multi-lag, rolling mean/std.
    Rolling'ler shift(1) ile yapılır (tamamen geçmişe dayalı).
    """
    panel = panel.sort_values(["Market_Index", "Date"]).reset_index(drop=True)

    panel["dow"] = panel["Date"].dt.dayofweek.astype(int)
    panel["month"] = panel["Date"].dt.month.astype(int)

    panel["ret_t0"] = panel["Index_Change_Percent"].astype(float)

    for lag in [1, 2, 3, 5]:
        panel[f"lag_ret_{lag}"] = (
            panel.groupby("Market_Index")["Index_Change_Percent"].shift(lag).fillna(0.0)
        )

    for w in [3, 5, 10, 20]:
        panel[f"roll_mean_{w}"] = panel.groupby("Market_Index")["Index_Change_Percent"].transform(
            lambda s: s.shift(1).rolling(window=w).mean()
        ).fillna(0.0)
        panel[f"roll_std_{w}"] = panel.groupby("Market_Index")["Index_Change_Percent"].transform(
            lambda s: s.shift(1).rolling(window=w).std()
        ).fillna(0.0)

    return panel


def fit_eval_once(panel: pd.DataFrame,
                  emb_matrix: np.ndarray,
                  numeric_cols: list[str],
                  cat_cols: list[str],
                  train_mask: np.ndarray,
                  val_mask: np.ndarray,
                  svd_dim: int = 64,
                  seed: int = 42,
                  thr_objective: str = "mcc"):
    """
    Tek bir train/val split üzerinde:
      - scaler + OHE + SVD(emb) fit sadece train'de
      - XGB train
      - threshold search (thr_objective: mcc/bacc)
    """
    X_numeric = panel[numeric_cols].values.astype(float)
    X_cat = panel[cat_cols].values
    y = panel["Target_Direction"].values.astype(int)

    Xn_tr, Xn_va = X_numeric[train_mask], X_numeric[val_mask]
    Xc_tr, Xc_va = X_cat[train_mask], X_cat[val_mask]
    Xe_tr, Xe_va = emb_matrix[train_mask], emb_matrix[val_mask]
    y_tr, y_va = y[train_mask], y[val_mask]

    scaler = StandardScaler()
    Xn_tr_s = scaler.fit_transform(Xn_tr)
    Xn_va_s = scaler.transform(Xn_va)

    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    Xc_tr_o = ohe.fit_transform(Xc_tr)
    Xc_va_o = ohe.transform(Xc_va)

    svd = TruncatedSVD(n_components=svd_dim, random_state=seed)
    Xe_tr_k = svd.fit_transform(Xe_tr)
    Xe_va_k = svd.transform(Xe_va)

    X_tr = np.concatenate([Xn_tr_s, Xc_tr_o, Xe_tr_k], axis=1)
    X_va = np.concatenate([Xn_va_s, Xc_va_o, Xe_va_k], axis=1)

    xgb_model = train_model.train_xgb(
        X_train=X_tr,
        y_train=y_tr,
        X_val=X_va,
        y_val=y_va
    )

    va_prob = xgb_model.predict_proba(X_va)[:, 1]
    pr_auc = float(average_precision_score(y_va, va_prob))
    roc_auc = float(roc_auc_score(y_va, va_prob))

    best = search_best_threshold(y_va, va_prob, objective=thr_objective)

    va_pred_best = (va_prob >= best["thr"]).astype(int)
    cm = confusion_matrix(y_va, va_pred_best)

    metrics = {
        "PR_AUC": pr_auc,
        "ROC_AUC": roc_auc,
        "BalancedAcc": float(best["bacc"]),
        "MCC": float(best["mcc"]),
        "thr": float(best["thr"]),
        "val_n": int(val_mask.sum()),
        "cm": cm,
    }

    artifacts = {
        "model": xgb_model,
        "scaler": scaler,
        "ohe": ohe,
        "svd": svd,
    }

    return metrics, artifacts, (y_va, va_pred_best)


def main():
    print("=" * 60)
    print("PROJE PIPELINE: XGBOOST + FINBERT (ROW-AGG) + DAILY PANEL + SVD + RETURN-DYNAMICS (NEXT-DAY)")
    print("=" * 60)

    # -----------------------------
    # Config (ENV)
    # -----------------------------
    SAVE_MODELS = os.getenv("SAVE_MODELS", "1") == "1"
    CUTOFF_FRAC = float(os.getenv("CUTOFF_FRAC", "0.80"))
    SVD_DIM = int(os.getenv("SVD_DIM", "64"))

    CV_MODE = os.getenv("CV_MODE", "0") == "1"
    CV_POINTS_STR = os.getenv("CV_POINTS", "0.60,0.70,0.80,0.90")
    CV_POINTS = [float(x.strip()) for x in CV_POINTS_STR.split(",") if x.strip()]

    K_VOL = float(os.getenv("K_VOL", "0.90"))
    MIN_THR = float(os.getenv("MIN_THR", "0.15"))

    THR_OBJECTIVE = os.getenv("THR_OBJECTIVE", "mcc").strip().lower()

    print(f"[CFG] SAVE_MODELS={int(SAVE_MODELS)} | CUTOFF_FRAC={CUTOFF_FRAC} | SVD_DIM={SVD_DIM}")
    print(f"[CFG] CV_MODE={int(CV_MODE)} | CV_POINTS={CV_POINTS}")
    print(f"[CFG] Adaptive labeling: K_VOL={K_VOL} | MIN_THR={MIN_THR}")
    print(f"[CFG] THR_OBJECTIVE={THR_OBJECTIVE}")

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

    # roll_std_10 güvenliği
    if "roll_std_10" not in panel.columns:
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

    if len(panel) < 300:
        print("[WARN] Çok az satır kaldı; sonuçlar oynak olabilir.")

    # daily embedding matrisi (768)
    emb_matrix = np.vstack(panel["emb_mean"].to_numpy()).astype(float)

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

    # -----------------------------
    # Walk-forward CV (optional)
    # -----------------------------
    unique_dates = np.sort(panel["Date"].unique())
    if len(unique_dates) < 30:
        print("[WARN] Unique date sayısı düşük; CV oynak olabilir.")

    if CV_MODE:
        print("\n[CV_MODE=1] Walk-forward evaluation running...")

        cut_idx = [max(1, min(len(unique_dates) - 2, int(len(unique_dates) * p))) for p in CV_POINTS]
        cut_dates = [unique_dates[i] for i in cut_idx]

        folds = []
        for i, cd in enumerate(cut_dates):
            if i < len(cut_dates) - 1:
                vd_end = cut_dates[i + 1]
                tr_mask = (panel["Date"] <= cd).to_numpy()
                va_mask = ((panel["Date"] > cd) & (panel["Date"] <= vd_end)).to_numpy()
            else:
                tr_mask = (panel["Date"] <= cd).to_numpy()
                va_mask = (panel["Date"] > cd).to_numpy()

            if va_mask.sum() < 50:
                print(f"[CV] Fold {i+1}: val too small ({int(va_mask.sum())}), skipping.")
                continue

            m, _, _ = fit_eval_once(
                panel=panel,
                emb_matrix=emb_matrix,
                numeric_cols=numeric_cols,
                cat_cols=cat_cols,
                train_mask=tr_mask,
                val_mask=va_mask,
                svd_dim=SVD_DIM,
                seed=42,
                thr_objective=THR_OBJECTIVE,
            )
            folds.append(m)

            print(
                f"[CV] Fold {i+1} | val_n={m['val_n']} | "
                f"PR_AUC={m['PR_AUC']:.4f} ROC_AUC={m['ROC_AUC']:.4f} "
                f"BAcc={m['BalancedAcc']:.4f} MCC={m['MCC']:.4f} thr={m['thr']:.2f}"
            )

        if not folds:
            raise RuntimeError("No valid folds produced. Check CV_POINTS / date distribution.")

        mean_pr = float(np.mean([f["PR_AUC"] for f in folds]))
        mean_roc = float(np.mean([f["ROC_AUC"] for f in folds]))
        mean_ba = float(np.mean([f["BalancedAcc"] for f in folds]))
        mean_mcc = float(np.mean([f["MCC"] for f in folds]))

        print("\n[CV] MEAN RESULTS")
        print(f"PR_AUC={mean_pr:.4f} | ROC_AUC={mean_roc:.4f} | BalancedAcc={mean_ba:.4f} | MCC={mean_mcc:.4f}")
        print("[CV] Done. (CV_MODE=1 -> model kaydı yapılmaz.)")
        return

    # -----------------------------
    # Single out-of-time split
    # -----------------------------
    cutoff_idx = max(1, min(len(unique_dates) - 2, int(len(unique_dates) * CUTOFF_FRAC)))
    cutoff_date = unique_dates[cutoff_idx]
    train_mask = (panel["Date"] <= cutoff_date).to_numpy()
    val_mask = (panel["Date"] > cutoff_date).to_numpy()

    print(f"\n[Single Split] cutoff_frac={CUTOFF_FRAC:.2f} cutoff_date={cutoff_date}")
    if val_mask.sum() < 50:
        print(f"[WARN] Validation set küçük (n={int(val_mask.sum())}).")

    print("[6/7] Scaling + OHE + SVD + concat...")
    metrics, artifacts, (y_val, y_pred) = fit_eval_once(
        panel=panel,
        emb_matrix=emb_matrix,
        numeric_cols=numeric_cols,
        cat_cols=cat_cols,
        train_mask=train_mask,
        val_mask=val_mask,
        svd_dim=SVD_DIM,
        seed=42,
        thr_objective=THR_OBJECTIVE,
    )

    print(f"[Validation] PR-AUC (Average Precision): {metrics['PR_AUC']:.4f}")
    print(f"[Validation] ROC-AUC                : {metrics['ROC_AUC']:.4f}")
    print(
        f"[Threshold Search:{THR_OBJECTIVE}] Best threshold: {metrics['thr']:.2f} | "
        f"BalancedAcc={metrics['BalancedAcc']:.4f} | MCC={metrics['MCC']:.4f}"
    )

    print("\n[Validation] Confusion Matrix (best thr)")
    print(metrics["cm"])
    print("\n[Validation] Classification Report (best thr)")
    print(classification_report(y_val, y_pred, target_names=["Down", "Up"]))

    # -----------------------------
    # Save artifacts (optional)
    # -----------------------------
    if SAVE_MODELS:
        os.makedirs("models", exist_ok=True)
        joblib.dump(float(metrics["thr"]), "models/threshold.pkl")
        joblib.dump(artifacts["model"], "models/xgb_model.pkl")
        joblib.dump(artifacts["scaler"], "models/scaler.pkl")
        joblib.dump(numeric_cols, "models/numeric_cols.pkl")
        joblib.dump(cat_cols, "models/cat_cols.pkl")
        joblib.dump(artifacts["ohe"], "models/ohe.pkl")
        joblib.dump(artifacts["svd"], "models/svd.pkl")

        print("\nBAŞARILI! Model kaydedildi.")
        print("models/xgb_model.pkl, models/scaler.pkl, models/ohe.pkl, models/svd.pkl, models/threshold.pkl")
    else:
        print("\n[SAVE_MODELS=0] Model kaydı atlandı.")


if __name__ == "__main__":
    main()
