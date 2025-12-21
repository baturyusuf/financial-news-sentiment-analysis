# Dosya: main.py
import nltk

nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Modüller
from src.data import make_dataset
from src.features import build_features
from src.models import train_model


def main():
    print("=" * 60)
    print("PROJE PIPELINE: XGBOOST + FINBERT (NO PCA)")
    print("=" * 60)

    # === ADIM 1: Veri Yükleme ===
    print("[1/6] Veri yükleniyor...")
    raw_df = make_dataset.load_raw_data(
        'data/raw/financial_news_market_events_2025.csv'
    )
    processed_df = make_dataset.preprocess_data(raw_df).reset_index(drop=True)

    # === ADIM 2: Özellik Mühendisliği ===
    print("[2/6] Teknik İndikatörler Hazırlanıyor...")

    processed_df = build_features.add_commodity_features(
        processed_df, date_col="Date"
    )
    processed_df = build_features.add_technical_indicators(processed_df)
    processed_df = build_features.create_lagged_features(
        processed_df, 'Index_Change_Percent'
    )
    processed_df = build_features.add_stock_mention_feature(
        processed_df, out_col="mentions_stock"
    )
    processed_df = build_features.add_company_mention_feature(processed_df)

    processed_df = processed_df.dropna().reset_index(drop=True)

    # =========================
    # TARGET (LABEL) OLUŞTURMA
    # =========================
    processed_df['return_1d'] = processed_df['Index_Change_Percent'].shift(-1)
    processed_df = processed_df.dropna(subset=['return_1d'])

    THRESHOLD = 0.2  # %0.2
    processed_df['Target_Direction'] = (
        processed_df['return_1d'] > THRESHOLD
    ).astype(int)

    print(f"Eğitime girecek satır sayısı: {len(processed_df)}")
    print(processed_df['Target_Direction'].value_counts(normalize=True))

    # === ADIM 3: FinBERT Embedding + Sentiment ===
    print("[3/6] Embeddingler hesaplanıyor...")
    extractor = build_features.FinbertFeatureExtractor()

    headlines = processed_df['Headline'].tolist()
    embeddings = []
    sentiments = []

    total = len(headlines)
    for i, text in enumerate(headlines):
        if i % 200 == 0:
            print(f"  Embedding: {i}/{total}")
        embeddings.append(extractor.get_embedding(text))
        sentiments.append(extractor.get_sentiment_scores(text))

    embeddings = np.array(embeddings)

    processed_df['sent_score'] = [
        s['positive'] - s['negative'] for s in sentiments
    ]
    processed_df['sent_conf'] = [
        max(s.values()) for s in sentiments
    ]

    print(
        "SENTIMENT KONTROL:",
        processed_df[['sent_score', 'sent_conf']].describe()
    )

    # === FEATURE MATRİSİ ===
    numeric_cols = [
        'Trading_Volume',
        'Previous_Index_Change_Percent_1d',
        'Volatility_7d',
        'RSI_14',
        'Cumulative_Return_3d',
        'Gold_close_t1',
        'Gold_ret1_t1',
        'Oil_close_t1',
        'Oil_ret1_t1',
        'keyword_bullish',
        'keyword_bearish',
        'mentions_stock',
        'sent_score',   # ✅ EKLENDİ
        'sent_conf',    # ✅ EKLENDİ
    ]

    for col in numeric_cols:
        if col not in processed_df.columns:
            processed_df[col] = 0

    X_numeric = processed_df[numeric_cols].values
    y = processed_df['Target_Direction'].values


    # === KRONOLOJİK SPLIT ===
    split_idx = int(len(processed_df) * 0.8)

    X_num_train = X_numeric[:split_idx]
    X_emb_train = embeddings[:split_idx]
    y_train = y[:split_idx]

    X_num_val = X_numeric[split_idx:]
    X_emb_val = embeddings[split_idx:]
    y_val = y[split_idx:]

    # === ADIM 5: Scaling + Birleştirme ===
    print("[5/6] Scaling + Embedding Birleştirme...")

    scaler = StandardScaler()
    X_num_train_scaled = scaler.fit_transform(X_num_train)
    X_num_val_scaled = scaler.transform(X_num_val)

    X_train_final = np.concatenate(
        [X_num_train_scaled, X_emb_train], axis=1
    )
    X_val_final = np.concatenate(
        [X_num_val_scaled, X_emb_val], axis=1
    )

    print(f"Eğitim Matrisi Son Boyutu: {X_train_final.shape}")

    # === ADIM 6: Model Eğitimi ===
    print("[6/6] Model Eğitiliyor...")

    xgb_model = train_model.train_xgb(
        X_train=X_train_final,
        y_train=y_train,
        X_val=X_val_final,
        y_val=y_val
    )

    os.makedirs('models', exist_ok=True)
    joblib.dump(xgb_model, 'models/xgb_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')

    print("\nBAŞARILI! Model kaydedildi.")
    print("models/xgb_model.pkl, models/scaler.pkl")


if __name__ == "__main__":
    main()
