# Dosya: src/inference_xgb.py
import joblib
import numpy as np
import pandas as pd

from src.features import build_features


def load_artifacts(models_dir: str = "models"):
    model = joblib.load(f"{models_dir}/xgb_model.pkl")
    scaler = joblib.load(f"{models_dir}/scaler.pkl")
    ohe = joblib.load(f"{models_dir}/ohe.pkl")
    svd = joblib.load(f"{models_dir}/svd.pkl")
    thr = float(joblib.load(f"{models_dir}/threshold.pkl"))

    numeric_cols = joblib.load(f"{models_dir}/numeric_cols.pkl")
    cat_cols = joblib.load(f"{models_dir}/cat_cols.pkl")

    return {
        "model": model,
        "scaler": scaler,
        "ohe": ohe,
        "svd": svd,
        "thr": thr,
        "numeric_cols": numeric_cols,
        "cat_cols": cat_cols,
    }


def finbert_day_aggregate(headlines: list[str]):
    """
    Eğitimdeki ile aynı mantık:
      - her headline -> FinBERT embedding(768) + sentiment
      - mean pooling + sent mean + conf max vb.
    """
    extractor = build_features.FinbertFeatureExtractor()

    if not headlines:
        # haber yoksa: embedding=0, sentiment=0
        emb_mean = np.zeros((768,), dtype=float)
        agg = {
            "News_Count": 0,
            "sent_pos_mean": 0.0, "sent_neg_mean": 0.0, "sent_neu_mean": 0.0,
            "sent_conf_mean": 0.0, "sent_conf_max": 0.0,
            "sent_score_mean": 0.0, "sent_score_max": 0.0, "sent_score_min": 0.0, "sent_score_std": 0.0,
            "emb_std_mean": 0.0,
        }
        return emb_mean, agg

    emb_list = []
    pos_list, neg_list, neu_list, conf_list = [], [], [], []

    for t in headlines:
        t = "" if t is None else str(t)
        e = extractor.get_embedding(t)  # (768,)
        emb_list.append(e)

        s = extractor.get_sentiment_scores(t)
        s_lower = {str(k).lower(): float(v) for k, v in s.items()} if isinstance(s, dict) else {}
        pos_list.append(s_lower.get("positive", 0.0))
        neg_list.append(s_lower.get("negative", 0.0))
        neu_list.append(s_lower.get("neutral", 0.0))
        conf_list.append(max(s_lower.values()) if len(s_lower) else 0.0)

    emb_block = np.asarray(emb_list, dtype=float)
    emb_mean = emb_block.mean(axis=0)
    emb_std_mean = float(emb_block.std(axis=0).mean()) if len(emb_block) > 1 else 0.0

    pos = np.asarray(pos_list, dtype=float)
    neg = np.asarray(neg_list, dtype=float)
    neu = np.asarray(neu_list, dtype=float)
    conf = np.asarray(conf_list, dtype=float)

    sent_score = pos - neg

    agg = {
        "News_Count": int(len(headlines)),
        "sent_pos_mean": float(pos.mean()),
        "sent_neg_mean": float(neg.mean()),
        "sent_neu_mean": float(neu.mean()),
        "sent_conf_mean": float(conf.mean()),
        "sent_conf_max": float(conf.max()),
        "sent_score_mean": float(sent_score.mean()),
        "sent_score_max": float(sent_score.max()),
        "sent_score_min": float(sent_score.min()),
        "sent_score_std": float(sent_score.std()) if len(sent_score) > 1 else 0.0,
        "emb_std_mean": float(emb_std_mean),
    }
    return emb_mean, agg


def build_single_day_row(
    date: str | pd.Timestamp,
    market_index: str,
    index_change_percent: float,
    trading_volume: float,
    market_event: str = "UNKNOWN",
    sector: str = "UNKNOWN",
    impact_level: str = "UNKNOWN",
    source: str = "UNKNOWN",
    keyword_bullish: int = 0,
    keyword_bearish: int = 0,
    mentions_stock: int = 0,
    headlines: list[str] | None = None,
):
    """
    Streamlit tarafında tek gün için model inputu üretmek için 'ham' satır.
    Teknik/commodity/rolling feature'lar bu tek satırdan üretilemez;
    deploy’da bunları üretmek için mutlaka geçmiş günleri de içeren panel gerekir.
    Bu fonksiyon, en azından haber agregasyonu + temel alanları hazırlar.
    """
    if headlines is None:
        headlines = []

    emb_mean, sent_agg = finbert_day_aggregate(headlines)

    row = {
        "Date": pd.to_datetime(date).tz_localize(None),
        "Market_Index": str(market_index),

        "Index_Change_Percent": float(index_change_percent),
        "Trading_Volume": float(trading_volume),

        "Market_Event": str(market_event),
        "Sector": str(sector),
        "Impact_Level": str(impact_level),
        "Source": str(source),

        "keyword_bullish": int(keyword_bullish),
        "keyword_bearish": int(keyword_bearish),
        "mentions_stock": int(mentions_stock),

        **sent_agg,
        "emb_mean": emb_mean,
    }
    return row


def predict_from_panel_row(panel_row: dict, artifacts: dict, history_panel: pd.DataFrame | None = None):
    """
    Eğitimle birebir uyum için önerilen kullanım:
      - history_panel: geçmiş günleri içeren panel (Index_Change_Percent, Trading_Volume, ... ve haber agregasyonları)
      - panel_row: bugünün row'u (build_single_day_row ile üret)
      - sonra concat edip eğitimdeki gibi commodity + technical + return_dynamics hesaplayıp en son günü predict etmek.

    Eğer history_panel None verilirse: sadece mevcut row’dan bazı feature’lar eksik kalır -> sonuç güvenilmez.
    """
    model = artifacts["model"]
    scaler = artifacts["scaler"]
    ohe = artifacts["ohe"]
    svd = artifacts["svd"]
    thr = artifacts["thr"]

    numeric_cols = artifacts["numeric_cols"]
    cat_cols = artifacts["cat_cols"]

    if history_panel is None:
        raise ValueError("history_panel gerekli. Tek satırdan rolling/technical/commodity feature üretilemez.")

    # history + today
    today_df = pd.DataFrame([panel_row])
    panel = pd.concat([history_panel.copy(), today_df], ignore_index=True)
    panel["Date"] = pd.to_datetime(panel["Date"]).dt.tz_localize(None)

    # eğitimdeki gibi commodity + technical + lag/rolling üret
    panel = build_features.add_commodity_features(panel, date_col="Date")

    out = []
    for midx, g in panel.groupby("Market_Index", sort=False):
        g = g.sort_values("Date").reset_index(drop=True)
        g = build_features.add_technical_indicators(g, price_col="Index_Change_Percent")
        out.append(g)
    panel = pd.concat(out, ignore_index=True).sort_values(["Market_Index", "Date"]).reset_index(drop=True)

    panel["Previous_Index_Change_Percent_1d"] = (
        panel.groupby("Market_Index")["Index_Change_Percent"].shift(1).fillna(0.0)
    )

    # return dynamics (aynı mantık)
    panel["dow"] = panel["Date"].dt.dayofweek.astype(int)
    panel["month"] = panel["Date"].dt.month.astype(int)
    panel["ret_t0"] = panel["Index_Change_Percent"].astype(float)

    for lag in [1, 2, 3, 5]:
        panel[f"lag_ret_{lag}"] = panel.groupby("Market_Index")["Index_Change_Percent"].shift(lag).fillna(0.0)

    for w in [3, 5, 10, 20]:
        panel[f"roll_mean_{w}"] = panel.groupby("Market_Index")["Index_Change_Percent"].transform(
            lambda s: s.shift(1).rolling(window=w).mean()
        ).fillna(0.0)
        panel[f"roll_std_{w}"] = panel.groupby("Market_Index")["Index_Change_Percent"].transform(
            lambda s: s.shift(1).rolling(window=w).std()
        ).fillna(0.0)

    # son satır = tahmin edilecek gün (today)
    panel = panel.sort_values(["Market_Index", "Date"]).reset_index(drop=True)
    last = panel.iloc[[-1]].copy()

    # eksik feature varsa 0/UNKNOWN ile doldur
    for c in numeric_cols:
        if c not in last.columns:
            last[c] = 0.0
    for c in cat_cols:
        if c not in last.columns:
            last[c] = "UNKNOWN"
    last[cat_cols] = last[cat_cols].fillna("UNKNOWN").astype(str)

    emb_matrix = np.vstack(last["emb_mean"].to_numpy()).astype(float)

    X_num = last[numeric_cols].values.astype(float)
    X_cat = last[cat_cols].values

    X_num_s = scaler.transform(X_num)
    X_cat_o = ohe.transform(X_cat)
    X_emb_k = svd.transform(emb_matrix)

    X = np.concatenate([X_num_s, X_cat_o, X_emb_k], axis=1)

    prob_up = float(model.predict_proba(X)[:, 1][0])
    pred = int(prob_up >= thr)

    return {
        "prob_up": prob_up,
        "thr": thr,
        "pred": pred,  # 1=Up, 0=Down
    }
