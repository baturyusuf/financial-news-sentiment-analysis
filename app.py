# Dosya: app.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import joblib
import numpy as np
import torch
from src.features import build_features
from src.utils_live import get_stock_data, train_and_predict_lstm, get_news_for_ticker

st.set_page_config(page_title="FinAI - Sentiment & Prediction", layout="wide")


# === CACHE MEKANİZMASI ===
@st.cache_resource
@st.cache_resource
def load_resources():
    try:
        model = joblib.load('models/xgb_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        numeric_cols = joblib.load('models/numeric_cols.pkl')
        cat_cols = joblib.load('models/cat_cols.pkl')
        ohe = joblib.load('models/ohe.pkl')
        svd = joblib.load('models/svd.pkl')
        threshold = joblib.load('models/threshold.pkl')
        extractor = build_features.FinbertFeatureExtractor()
        return model, scaler, extractor, numeric_cols, cat_cols, ohe, svd, threshold
    except Exception as e:
        st.error(f"Model dosyaları eksik! Önce 'python main.py' çalıştırın. Hata: {e}")
        return None, None, None, None, None, None, None, None

xgb_model, scaler, feature_extractor, numeric_cols, cat_cols, ohe, svd, threshold = load_resources()

# === SIDEBAR ===
st.sidebar.title("FinAI Panel")
ticker_input = st.sidebar.text_input("Hisse Kodu", value="BTC-USD")
period_input = st.sidebar.selectbox("Veri Aralığı", ["1mo", "6mo", "1y", "2y", "5y"], index=3)

# === ANA EKRAN ===
st.title(f"🔍 Finansal Analiz: {ticker_input}")

df, ticker_obj = get_stock_data(ticker_input, period=period_input)

if df.empty:
    st.error("Veri çekilemedi. Kodu kontrol edin.")
else:
    tab1, tab2 = st.tabs(["📈 Teknik & LSTM", "📰 Haber & RF (PCA)"])

    # --- TAB 1: LSTM (Aynı kalıyor) ---
    with tab1:
        col1, col2 = st.columns([3, 1])
        with col1:
            fig = go.Figure(
                data=[go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'])])
            fig.update_layout(height=500, title=f"{ticker_input} Fiyat Grafiği")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("LSTM Tahmini")
            current_price = df['Close'].iloc[-1]
            st.metric("Son Fiyat", f"${current_price:.2f}")

            if st.button("LSTM Analizi Yap"):
                with st.spinner("LSTM Eğitiliyor..."):
                    pred = train_and_predict_lstm(df)
                    delta = ((pred - current_price) / current_price) * 100
                    st.metric("Yarınki Tahmin", f"${pred:.2f}", f"{delta:.2f}%")

        # --- TAB 2: HABER & RF (GÜNCELLENMİŞ - LİSTELEMELİ) ---
        with tab2:
            st.subheader("Haber Analizi (Random Forest + PCA)")

            if st.button("Haberleri Analiz Et"):
                with st.spinner("Haberler çekiliyor ve işleniyor..."):
                    news_list = get_news_for_ticker(ticker_obj)

                    if not news_list:
                        st.warning("Haber bulunamadı.")
                    else:
                        st.success(f"{len(news_list)} haber bulundu.")

                        # --- BÖLÜM 1: TÜM HABERLERİ LİSTELE ---
                        st.markdown("### 📢 Son Gelişmeler")
                        for i, news in enumerate(news_list):
                            st.markdown(f"**{i + 1}.** [{news['title']}]({news['link']})")
                            st.caption(f"Kaynak: {news['publisher']}")

                        st.divider()

                        # --- BÖLÜM 2: MODEL TAHMİNİ (EN GÜNCEL HABER İLE) ---
                        # Model tahmini için piyasanın en sıcak bilgisini (ilk haberi) kullanıyoruz.
                        latest_news = news_list[0]

                        st.markdown(f"### 🧠 Analiz Odaklı Haber")
                        st.info(
                            f"Model, piyasa yönünü tahmin etmek için en güncel haberi ({latest_news['title']}) ve teknik verileri kullanıyor.")

                        # 2. Embedding ve Sentiment
                        # raw_embedding = feature_extractor.get_embedding(latest_news['title'])
                        sent_scores = feature_extractor.get_sentiment_scores(latest_news['title'])
                        sentiment_label = max(sent_scores, key=sent_scores.get)

                        # 3. PCA İŞLEMİ
                        embedding = feature_extractor.get_embedding(latest_news['title']).reshape(1, -1)

                        # 4. Teknik Verileri Hazırla
                        processed_df = df.copy().reset_index()
                        processed_df = processed_df.rename(columns={"Close": "Index_Change_Percent"})
                        processed_df['Index_Change_Percent'] = processed_df['Index_Change_Percent'].pct_change() * 100

                        processed_df = build_features.add_technical_indicators(processed_df)
                        processed_df = build_features.add_commodity_features(processed_df, date_col="Date")

                        latest_data = processed_df.iloc[[-1]].copy()

                        # Feature'ları Doldur
                        latest_data['keyword_bullish'] = 0
                        latest_data['keyword_bearish'] = 0
                        latest_data['mentions_stock'] = 1
                        latest_data['Previous_Index_Change_Percent_1d'] = processed_df['Index_Change_Percent'].iloc[-2]


                        # for c in cols:
                        #     if c not in latest_data.columns: latest_data[c] = 0
                        #
                        # # 5. Birleştirme
                        # X_numeric = latest_data[cols].values
                        # X_numeric_scaled = scaler.transform(X_numeric)
                        #
                        # X_final = np.concatenate([X_numeric_scaled, embedding], axis=1)

                        # 6. Tahmin
                        st.caption(f"Decision threshold: {float(threshold):.2f}")

                        # --- 0) yfinance Volume -> Trading_Volume hizalama ---
                        if "Volume" in processed_df.columns and "Trading_Volume" not in processed_df.columns:
                            processed_df["Trading_Volume"] = processed_df["Volume"]

                        # --- 1) Keyword/mentions feature'larını eğitimdeki mantıkla üret ---
                        tmp = pd.DataFrame({"Headline": [latest_news["title"]]})
                        tmp = build_features.add_stock_mention_feature(tmp, headline_col="Headline")
                        latest_data["keyword_bullish"] = int(tmp.loc[0, "keyword_bullish"])
                        latest_data["keyword_bearish"] = int(tmp.loc[0, "keyword_bearish"])
                        latest_data["mentions_stock"] = int(tmp.loc[0, "mentions_stock"])

                        # --- 2) Sentiment feature'larını eğitimdeki gibi üret: sent_score + sent_conf ---
                        sent_score = sent_scores.get("positive", 0.0) - sent_scores.get("negative", 0.0)
                        sent_conf = max(sent_scores.values()) if sent_scores else 0.0
                        latest_data["sent_score"] = float(sent_score)
                        latest_data["sent_conf"] = float(sent_conf)

                        # --- 3) Eksik kolonları 0 ile tamamla (eğitim feature listesine göre) ---
                        for c in numeric_cols:
                            if c not in latest_data.columns:
                                latest_data[c] = 0

                        # --- 4) Sıralama birebir eğitimle aynı: numeric_cols ---
                        X_numeric = latest_data[numeric_cols].values
                        X_numeric_scaled = scaler.transform(X_numeric)

                        # Embedding (1536) + numeric (14) -> 1550
                        # --- 0) yfinance Volume -> Trading_Volume hizalama ---
                        if "Volume" in processed_df.columns and "Trading_Volume" not in processed_df.columns:
                            processed_df["Trading_Volume"] = processed_df["Volume"]

                        # --- 1) keyword/mentions ---
                        tmp = pd.DataFrame({"Headline": [latest_news["title"]]})
                        tmp = build_features.add_stock_mention_feature(tmp, headline_col="Headline")

                        latest_data["keyword_bullish"] = int(tmp.loc[0, "keyword_bullish"])
                        latest_data["keyword_bearish"] = int(tmp.loc[0, "keyword_bearish"])
                        latest_data["mentions_stock"] = int(tmp.loc[0, "mentions_stock"])

                        # --- 2) sentiment aggregation (tek haber -> mean=max=min, std=0) ---
                        sent_scores = feature_extractor.get_sentiment_scores(latest_news["title"])
                        s_lower = {str(k).lower(): float(v) for k, v in sent_scores.items()} if isinstance(sent_scores,
                                                                                                           dict) else {}
                        pos = s_lower.get("positive", 0.0)
                        neg = s_lower.get("negative", 0.0)
                        neu = s_lower.get("neutral", 0.0)
                        conf = max(s_lower.values()) if len(s_lower) else 0.0
                        score = pos - neg

                        latest_data["sent_pos_mean"] = float(pos)
                        latest_data["sent_neg_mean"] = float(neg)
                        latest_data["sent_neu_mean"] = float(neu)
                        latest_data["sent_conf_mean"] = float(conf)
                        latest_data["sent_conf_max"] = float(conf)
                        latest_data["sent_score_mean"] = float(score)
                        latest_data["sent_score_max"] = float(score)
                        latest_data["sent_score_min"] = float(score)
                        latest_data["sent_score_std"] = 0.0

                        # News_Count (tek haber)
                        latest_data["News_Count"] = 1

                        # --- 3) Embedding (768) -> SVD(64) ---
                        emb_768 = feature_extractor.get_embedding(latest_news["title"]).reshape(1, -1)
                        emb_64 = svd.transform(emb_768)

                        # --- 4) Numeric kolonları tamamla ---
                        for c in numeric_cols:
                            if c not in latest_data.columns:
                                latest_data[c] = 0.0

                        X_num = latest_data[numeric_cols].values.astype(float)
                        X_num_scaled = scaler.transform(X_num)

                        # --- 5) Categoricals (demo: UNKNOWN) ---
                        cat_row = pd.DataFrame({c: ["UNKNOWN"] for c in cat_cols}).values
                        X_cat_ohe = ohe.transform(cat_row)

                        # --- 6) Final concat ---
                        X_final = np.concatenate([X_num_scaled, X_cat_ohe, emb_64], axis=1)

                        # --- 7) Predict ---
                        p_up = float(xgb_model.predict_proba(X_final)[0, 1])
                        pred = 1 if p_up >= float(threshold) else 0

                        # --- 5) Tahmin: threshold ile karar ver ---
                        col_res1, col_res2 = st.columns(2)

                        with col_res1:
                            color = (
                                "green" if sentiment_label == "positive"
                                else "red" if sentiment_label == "negative"
                                else "gray"
                            )
                            st.metric(
                                "Haber Duygusu",
                                sentiment_label.upper(),
                                f"pos={sent_scores['positive']:.2f} | neg={sent_scores['negative']:.2f}"
                            )

                        with col_res2:
                            if pred == 1:
                                st.success(f"Yön Tahmini: YÜKSELİŞ (P(Up)=%{p_up * 100:.1f})")
                            else:
                                st.error(f"Yön Tahmini: DÜŞÜŞ (P(Up)=%{p_up * 100:.1f})")