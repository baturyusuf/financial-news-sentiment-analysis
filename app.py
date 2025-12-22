# Dosya: app.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import joblib
import numpy as np

from src.features import build_features
from src.utils_live import get_stock_data, train_and_predict_lstm, get_news_for_ticker

st.set_page_config(page_title="FinAI - Sentiment & Prediction", layout="wide")


# =========================
# CACHE / LOAD MODELS
# =========================
@st.cache_resource
def load_resources():
    try:
        model = joblib.load("models/xgb_model.pkl")
        scaler = joblib.load("models/scaler.pkl")
        numeric_cols = joblib.load("models/numeric_cols.pkl")
        cat_cols = joblib.load("models/cat_cols.pkl")
        ohe = joblib.load("models/ohe.pkl")
        svd = joblib.load("models/svd.pkl")
        threshold = joblib.load("models/threshold.pkl")
        extractor = build_features.FinbertFeatureExtractor()
        return model, scaler, extractor, numeric_cols, cat_cols, ohe, svd, float(threshold)
    except Exception as e:
        st.error(f"Model dosyaları eksik! Önce 'python main.py' çalıştırın. Hata: {e}")
        return None, None, None, None, None, None, None, None


xgb_model, scaler, feature_extractor, numeric_cols, cat_cols, ohe, svd, threshold = load_resources()
if xgb_model is None:
    st.stop()


# =========================
# HELPERS
# =========================
def _safe_get_date_col(df_: pd.DataFrame) -> pd.DataFrame:
    """reset_index sonrası 'Date' yoksa 'index' -> 'Date' yap."""
    if "Date" not in df_.columns and "index" in df_.columns:
        df_ = df_.rename(columns={"index": "Date"})
    return df_


def finbert_sentiment_details(text: str) -> dict:
    """
    FinBERT'ten çıkan sözlüğü normalize eder:
      - pos/neg/neu
      - label
      - conf=max(prob)
      - score=pos-neg
    """
    raw = feature_extractor.get_sentiment_scores(text)
    s = {str(k).lower(): float(v) for k, v in raw.items()} if isinstance(raw, dict) else {}
    pos = float(s.get("positive", 0.0))
    neg = float(s.get("negative", 0.0))
    neu = float(s.get("neutral", 0.0))
    conf = float(max(s.values())) if len(s) else 0.0

    # Etiket seçimi (key isimleri tutarlı değilse güvenli)
    if len(s):
        label = max(s, key=s.get)
    else:
        label = "unknown"

    score = pos - neg
    return {
        "pos": pos,
        "neg": neg,
        "neu": neu,
        "conf": conf,
        "score": float(score),
        "label": str(label).upper(),
        "raw": raw,
    }


def make_prediction_date(last_ts: pd.Timestamp) -> str:
    """
    Tahmin ufku t+1: pratikte 'sonraki gün'.
    Hisse senetlerinde hafta sonu/ tatil var; burada basitçe +1 gün yazıyoruz.
    (İstersen daha sonra trading calendar ile iyileştiririz.)
    """
    try:
        last_ts = pd.to_datetime(last_ts)
        target = (last_ts + pd.Timedelta(days=1)).date()
        return str(target)
    except Exception:
        return "t+1"


def uncertainty_proxy(p_up: float, thr: float, band: float = 0.10) -> float:
    """
    İkili modelde 'sabit' sınıfı yok.
    Bu değer: olasılığın threshold'a yakınlığından türetilmiş 'kararsızlık' proxy'si.
    - 1.0: tam threshold üstünde (maks kararsız)
    - 0.0: threshold'dan band kadar veya daha uzak (daha net)
    """
    if band <= 0:
        return 0.0
    d = abs(p_up - thr)
    u = max(0.0, 1.0 - (d / band))
    return float(min(1.0, u))


# =========================
# SIDEBAR
# =========================
st.sidebar.title("FinAI Panel")
ticker_input = st.sidebar.text_input("Hisse Kodu", value="BTC-USD")
period_input = st.sidebar.selectbox("Veri Aralığı", ["1mo", "6mo", "1y", "2y", "5y"], index=3)

# İsteğe bağlı: kararsızlık bandı (UI'dan ayarlanabilir)
st.sidebar.markdown("### Eşik Yakınlığı (Kararsızlık)")
unc_band = st.sidebar.slider("Kararsızlık bandı (|P(Up)-thr| < band)", 0.01, 0.30, 0.10, 0.01)

# =========================
# MAIN
# =========================
st.title(f"Finansal Analiz: {ticker_input}")

df, ticker_obj = get_stock_data(ticker_input, period=period_input)

if df is None or df.empty:
    st.error("Veri çekilemedi. Kodu kontrol edin.")
    st.stop()

tab1, tab2 = st.tabs(["📈 Teknik & LSTM", "📰 Haber & XGB (SVD)"])


# =========================
# TAB 1: TECH + LSTM
# =========================
with tab1:
    col1, col2 = st.columns([3, 1])
    with col1:
        fig = go.Figure(
            data=[go.Candlestick(
                x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"]
            )]
        )
        fig.update_layout(height=500, title=f"{ticker_input} Fiyat Grafiği")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("LSTM Tahmini")
        current_price = float(df["Close"].iloc[-1])
        st.metric("Son Fiyat", f"${current_price:.2f}")

        # Ufuk bilgisi
        last_ts = df.index[-1]
        st.caption(f"Tahmin ufku: t+1 (sonraki gün). Son veri tarihi: {pd.to_datetime(last_ts).date()}")

        if st.button("LSTM Analizi Yap"):
            with st.spinner("LSTM eğitiliyor..."):
                pred = float(train_and_predict_lstm(df))
                delta = ((pred - current_price) / current_price) * 100.0
                st.metric("t+1 Tahmin (LSTM)", f"${pred:.2f}", f"{delta:.2f}%")


# =========================
# TAB 2: NEWS + XGB
# =========================
with tab2:
    st.subheader("Haber Analizi (FinBERT Sentiment + XGBoost)")

    if st.button("Haberleri Analiz Et"):
        with st.spinner("Haberler çekiliyor ve işleniyor..."):
            news_list = get_news_for_ticker(ticker_obj)

        if not news_list:
            st.warning("Haber bulunamadı.")
            st.stop()

        st.success(f"{len(news_list)} haber bulundu.")

        # --- BÖLÜM 1: HABERLER + SENTIMENT (TÜMÜ) ---
        st.markdown("### Son Gelişmeler (Haber + Sentiment)")

        if "sent_cache" not in st.session_state:
            st.session_state["sent_cache"] = {}

        rows = []
        prog = st.progress(0)
        for i, news in enumerate(news_list):
            title = str(news.get("title", "")).strip()
            link = str(news.get("link", "")).strip()
            publisher = str(news.get("publisher", "UNKNOWN")).strip()

            if title in st.session_state["sent_cache"]:
                sdet = st.session_state["sent_cache"][title]
            else:
                sdet = finbert_sentiment_details(title)
                st.session_state["sent_cache"][title] = sdet

            rows.append({
                "No": i + 1,
                "Title": title,
                "Publisher": publisher,
                "Label": sdet["label"],
                "Pos": sdet["pos"],
                "Neg": sdet["neg"],
                "Neu": sdet["neu"],
                "Conf": sdet["conf"],
                "Score(pos-neg)": sdet["score"],
                "Link": link,
            })

            prog.progress(int(((i + 1) / len(news_list)) * 100))
        prog.empty()

        sent_df = pd.DataFrame(rows)

        # Tablo görünümü
        st.dataframe(
            sent_df[["No", "Publisher", "Label", "Pos", "Neg", "Neu", "Conf", "Score(pos-neg)", "Title"]],
            use_container_width=True,
            hide_index=True,
        )

        # Tek tek linkli liste (opsiyonel)
        with st.expander("Haberleri tek tek linkli gör (detay)"):
            for r in rows:
                st.markdown(f"**{r['No']}.** [{r['Title']}]({r['Link']})")
                st.caption(
                    f"Kaynak: {r['Publisher']} | "
                    f"Label={r['Label']} | "
                    f"pos={r['Pos']:.3f} neg={r['Neg']:.3f} neu={r['Neu']:.3f} "
                    f"| score(pos-neg)={r['Score(pos-neg)']:.3f} conf={r['Conf']:.3f}"
                )

        st.divider()

        # --- BÖLÜM 2: MODEL TAHMİNİ (EN GÜNCEL HABERLE) ---
        latest_news = news_list[0]
        latest_title = str(latest_news.get("title", "")).strip()

        st.markdown("### Model Tahmini (XGB)")
        st.info(
            f"Model, yön tahmini için en güncel haberi kullanır:\n\n{latest_title}"
        )

        # Sentiment detayları (en güncel haber)
        sdet_latest = st.session_state["sent_cache"].get(latest_title) or finbert_sentiment_details(latest_title)

        # Ufuk bilgisi
        last_ts = df.index[-1]
        target_date_str = make_prediction_date(last_ts)
        st.caption(
            f"Tahmin ufku: t+1 (sonraki gün / sonraki işlem günü). "
            f"Son veri tarihi: {pd.to_datetime(last_ts).date()} | Hedef: {target_date_str}"
        )

        # === Teknik verileri hazırlama ===
        processed_df = df.copy().reset_index()
        processed_df = _safe_get_date_col(processed_df)

        # Eğitim tarafındaki isimle uyum için: Close -> Index_Change_Percent (sonra pct_change)
        processed_df = processed_df.rename(columns={"Close": "Index_Change_Percent"})
        processed_df["Index_Change_Percent"] = processed_df["Index_Change_Percent"].pct_change() * 100.0

        # Volume -> Trading_Volume hizalama
        if "Volume" in processed_df.columns and "Trading_Volume" not in processed_df.columns:
            processed_df["Trading_Volume"] = processed_df["Volume"]

        # Teknik & emtia feature'ları (eğitimle daha uyumlu)
        processed_df = build_features.add_technical_indicators(processed_df, price_col="Index_Change_Percent")
        processed_df = build_features.add_commodity_features(processed_df, date_col="Date")

        latest_data = processed_df.iloc[[-1]].copy()

        # Previous_Index_Change_Percent_1d (lag1)
        if len(processed_df) >= 2:
            latest_data["Previous_Index_Change_Percent_1d"] = float(processed_df["Index_Change_Percent"].iloc[-2])
        else:
            latest_data["Previous_Index_Change_Percent_1d"] = 0.0

        # Keyword / mentions (tek haber)
        tmp = pd.DataFrame({"Headline": [latest_title]})
        tmp = build_features.add_stock_mention_feature(tmp, headline_col="Headline")
        latest_data["keyword_bullish"] = int(tmp.loc[0, "keyword_bullish"])
        latest_data["keyword_bearish"] = int(tmp.loc[0, "keyword_bearish"])
        latest_data["mentions_stock"] = int(tmp.loc[0, "mentions_stock"])

        # Sentiment aggregation (tek haber -> mean=max=min, std=0)
        latest_data["sent_pos_mean"] = float(sdet_latest["pos"])
        latest_data["sent_neg_mean"] = float(sdet_latest["neg"])
        latest_data["sent_neu_mean"] = float(sdet_latest["neu"])
        latest_data["sent_conf_mean"] = float(sdet_latest["conf"])
        latest_data["sent_conf_max"] = float(sdet_latest["conf"])

        latest_data["sent_score_mean"] = float(sdet_latest["score"])
        latest_data["sent_score_max"] = float(sdet_latest["score"])
        latest_data["sent_score_min"] = float(sdet_latest["score"])
        latest_data["sent_score_std"] = 0.0

        # News_Count (tek haber)
        latest_data["News_Count"] = 1

        # Embedding (768) -> SVD(64)
        emb_768 = feature_extractor.get_embedding(latest_title).reshape(1, -1)
        emb_64 = svd.transform(emb_768)

        # Eksik numeric kolonları 0 ile tamamla
        for c in numeric_cols:
            if c not in latest_data.columns:
                latest_data[c] = 0.0

        X_num = latest_data[numeric_cols].values.astype(float)
        X_num_scaled = scaler.transform(X_num)

        # Categoricals: demo olarak UNKNOWN (eğitimdeki cat_cols sırasıyla)
        cat_row = pd.DataFrame({c: ["UNKNOWN"] for c in cat_cols}).values
        X_cat_ohe = ohe.transform(cat_row)

        # Final concat
        X_final = np.concatenate([X_num_scaled, X_cat_ohe, emb_64], axis=1)

        # Predict: binary probs
        prob = xgb_model.predict_proba(X_final)[0]
        p_down = float(prob[0])
        p_up = float(prob[1])

        pred = 1 if p_up >= float(threshold) else 0

        # Kararsızlık proxy
        unc = uncertainty_proxy(p_up=p_up, thr=float(threshold), band=float(unc_band))

        # === UI: Sentiment + Prediction ===
        st.caption(f"Decision threshold: {float(threshold):.2f} | THR objective: (training config'e bağlı)")

        col_res1, col_res2 = st.columns(2)

        with col_res1:
            st.metric(
                "Haber Sentiment (FinBERT)",
                sdet_latest["label"],
                f"pos={sdet_latest['pos']:.3f} | neg={sdet_latest['neg']:.3f} | neu={sdet_latest['neu']:.3f} | "
                f"score(pos-neg)={sdet_latest['score']:.3f} | conf={sdet_latest['conf']:.3f}"
            )

        with col_res2:
            # Tüm skorlar
            st.markdown("**Sınıf Olasılıkları (XGB)**")
            st.write(f"P(Up):  {p_up*100:.1f}%")
            st.write(f"P(Down): {p_down*100:.1f}%")

            st.markdown("**Kararsızlık / Sabit Proxy**")
            st.write(f"Threshold’a yakınlık (proxy): {unc*100:.1f}%")
            st.caption(
                "Not: Model ikili sınıflandırma (Up/Down). 'Sabit' sınıfı eğitilmedi. "
                "Bu değer sadece P(Up) ile threshold arasındaki mesafeden türetilmiş bir yakınlık ölçüsüdür."
            )

            if pred == 1:
                st.success(f"Yön Tahmini (t+1): YÜKSELİŞ")
            else:
                st.error(f"Yön Tahmini (t+1): DÜŞÜŞ")
