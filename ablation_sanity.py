# ablation_sanity.py
import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    balanced_accuracy_score, matthews_corrcoef,
)

from src.data import make_dataset
from src.features import build_features


def topk_precision(y_true, prob, k_frac=0.10):
    k = max(1, int(len(y_true) * k_frac))
    idx = np.argsort(prob)[::-1][:k]
    return float(y_true[idx].mean())


def eval_metrics(y_true, prob, thr=0.5, name=""):
    pred = (prob >= thr).astype(int)
    return {
        "name": name,
        "ROC_AUC": float(roc_auc_score(y_true, prob)),
        "PR_AUC": float(average_precision_score(y_true, prob)),
        "BalancedAcc@thr": float(balanced_accuracy_score(y_true, pred)),
        "MCC@thr": float(matthews_corrcoef(y_true, pred)),
        "Top10%Precision": float(topk_precision(y_true, prob, 0.10)),
    }


def compute_or_load_finbert(df: pd.DataFrame, cache_dir="data/processed"):
    """
    df ile aynı uzunlukta embedding ve sentiment üretir.
    Cache varsa yükler; uzunluk uyuşmazsa cache'i yok sayıp yeniden üretir.
    """
    os.makedirs(cache_dir, exist_ok=True)
    emb_path = os.path.join(cache_dir, "finbert_emb_768.npy")
    sent_path = os.path.join(cache_dir, "finbert_sent.npy")

    n = len(df)

    if os.path.exists(emb_path) and os.path.exists(sent_path):
        try:
            embeddings = np.load(emb_path)
            sentiments = np.load(sent_path, allow_pickle=True)

            if len(embeddings) == n and len(sentiments) == n:
                print("Loading cached embeddings/sent (length matched).")
                return embeddings, sentiments
            else:
                print(
                    f"Cache length mismatch -> recompute. "
                    f"cache_emb={len(embeddings)}, cache_sent={len(sentiments)}, df={n}"
                )
        except Exception as e:
            print(f"Cache load failed -> recompute. Error: {e}")

    print("Computing embeddings/sent (fresh)...")
    extractor = build_features.FinbertFeatureExtractor()
    headlines = df["Headline"].astype(str).tolist()

    embeddings = []
    sentiments = []
    for i, text in enumerate(headlines):
        if i % 200 == 0:
            print(f"  {i}/{len(headlines)}")
        embeddings.append(extractor.get_embedding(text))  # 768 bekleniyor
        sentiments.append(extractor.get_sentiment_scores(text))

    embeddings = np.array(embeddings)
    sentiments = np.array(sentiments, dtype=object)

    np.save(emb_path, embeddings)
    np.save(sent_path, sentiments, allow_pickle=True)

    return embeddings, sentiments


def main(threshold=0.2, target_mode="next_day"):
    """
    target_mode:
      - "next_day": return_1d = shift(-1) (satır kaydırma)
      - "same_day": return = Index_Change_Percent (shift yok)
    """
    print("=" * 80)
    print("ABLATION + SANITY CHECKS")
    print("=" * 80)

    # 1) Load & preprocess
    raw_df = make_dataset.load_raw_data("data/raw/financial_news_market_events_2025.csv")
    df = make_dataset.preprocess_data(raw_df).reset_index(drop=True)

    df = build_features.add_commodity_features(df, date_col="Date")
    df = build_features.add_technical_indicators(df)
    df = build_features.create_lagged_features(df, "Index_Change_Percent")
    df = build_features.add_stock_mention_feature(df, out_col="mentions_stock")
    df = build_features.add_company_mention_feature(df)

    df = df.dropna().reset_index(drop=True)

    # 2) Target (neutral-drop)
    if target_mode == "next_day":
        df["return_1d"] = df["Index_Change_Percent"].shift(-1)
        df = df.dropna(subset=["return_1d"]).reset_index(drop=True)
        r = df["return_1d"]
    elif target_mode == "same_day":
        r = df["Index_Change_Percent"]
    else:
        raise ValueError("target_mode must be 'next_day' or 'same_day'")

    df["Target_Direction"] = np.where(
        r > threshold, 1,
        np.where(r < -threshold, 0, np.nan)
    )
    df = df.dropna(subset=["Target_Direction"]).reset_index(drop=True)
    df["Target_Direction"] = df["Target_Direction"].astype(int)

    print(f"Target mode: {target_mode} | Rows after neutral-drop: {len(df)}")
    print(df["Target_Direction"].value_counts(normalize=True))

    # 3) Embeddings + sentiments (cache-safe)
    embeddings, sentiments = compute_or_load_finbert(df)

    # 4) Sent features
    sent_score = np.array(
        [s.get("positive", 0.0) - s.get("negative", 0.0) for s in sentiments],
        dtype=float
    )
    sent_conf = np.array(
        [max(s.values()) if isinstance(s, dict) and len(s) else 0.0 for s in sentiments],
        dtype=float
    )

    if len(sent_score) != len(df) or len(sent_conf) != len(df):
        raise RuntimeError(
            f"Sent length mismatch after cache handling: "
            f"sent_score={len(sent_score)}, sent_conf={len(sent_conf)}, df={len(df)}"
        )

    df["sent_score"] = sent_score
    df["sent_conf"] = sent_conf

    # 5) Numeric cols (training ile aynı olmalı)
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
    ]

    exog_cols = [
        "Trading_Volume",
        "Gold_close_t1",
        "Gold_ret1_t1",
        "Oil_close_t1",
        "Oil_ret1_t1",
        "keyword_bullish",
        "keyword_bearish",
        "mentions_stock",
        "sent_score",
        "sent_conf",
    ]

    for c in numeric_cols:
        if c not in df.columns:
            df[c] = 0

    X_num = df[numeric_cols].values
    X_emb = embeddings
    y = df["Target_Direction"].values

    # 6) Chronological split
    split = int(len(df) * 0.8)
    Xn_tr, Xn_va = X_num[:split], X_num[split:]
    Xe_tr, Xe_va = X_emb[:split], X_emb[split:]
    y_tr, y_va = y[:split], y[split:]

    # Baseline
    majority = int(pd.Series(y_tr).value_counts().idxmax())
    maj_acc = float((y_va == majority).mean())
    print(f"\n[Baseline] Majority class={majority} | Accuracy={maj_acc:.4f}")

    # Scale numeric
    scaler = StandardScaler()
    Xn_tr_s = scaler.fit_transform(Xn_tr)
    Xn_va_s = scaler.transform(Xn_va)

    def run_lr(Xtr, Xva, tag):
        lr = LogisticRegression(max_iter=3000, n_jobs=-1)
        lr.fit(Xtr, y_tr)
        prob = lr.predict_proba(Xva)[:, 1]
        m = eval_metrics(y_va, prob, thr=0.5, name=tag)
        print("\n", m)
        return prob

    print("\n--- Logistic Regression Ablations ---")
    _ = run_lr(Xn_tr_s, Xn_va_s, "LR: numeric_only")
    # Exogenous-only numeric: Index_Change_Percent türevlerini çıkar
    X_exog = df[exog_cols].values
    Xex_tr, Xex_va = X_exog[:split], X_exog[split:]
    ex_scaler = StandardScaler()
    _ = run_lr(ex_scaler.fit_transform(Xex_tr), ex_scaler.transform(Xex_va), "LR: exogenous_numeric_only")

    # Sent-only (2 feature)
    sent_tr = df[["sent_score", "sent_conf"]].values[:split]
    sent_va = df[["sent_score", "sent_conf"]].values[split:]
    sent_scaler = StandardScaler()
    _ = run_lr(sent_scaler.fit_transform(sent_tr), sent_scaler.transform(sent_va), "LR: sent_score_conf_only")

    _ = run_lr(Xe_tr, Xe_va, "LR: embedding_only")
    _ = run_lr(np.hstack([Xn_tr_s, Xe_tr]), np.hstack([Xn_va_s, Xe_va]), "LR: numeric+embedding")

    print("\n--- SHUFFLE LABEL SANITY (LR: numeric+embedding) ---")
    y_tr_sh = y_tr.copy()
    np.random.shuffle(y_tr_sh)

    lr_sh = LogisticRegression(max_iter=3000, n_jobs=-1)
    lr_sh.fit(np.hstack([Xn_tr_s, Xe_tr]), y_tr_sh)
    prob_sh = lr_sh.predict_proba(np.hstack([Xn_va_s, Xe_va]))[:, 1]
    m_sh = eval_metrics(y_va, prob_sh, thr=0.5, name="LR shuffled-y: numeric+embedding")
    print("\n", m_sh)

    print("\nDONE.")


if __name__ == "__main__":
    # "same_day" veya "next_day" seçebilirsiniz:
    # main(threshold=0.2, target_mode="same_day")
    main(threshold=0.2, target_mode="same_day")
