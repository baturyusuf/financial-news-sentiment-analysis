# Dosya: main.py
import nltk

nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA  # <--- EKLENDİ
import joblib
import os

# Modüller
from src.data import make_dataset
from src.features import build_features
from src.models import train_model


def main():
    print("=" * 60)
    print("PROJE PIPELINE: PCA + RANDOM FOREST")
    print("=" * 60)

    # === ADIM 1: Veri Yükleme ===
    print("[1/6] Veri yükleniyor...")
    raw_df = make_dataset.load_raw_data('data/raw/financial_news_market_events_2025.csv')
    processed_df = make_dataset.preprocess_data(raw_df).reset_index(drop=True)

    # === ADIM 2: Özellik Mühendisliği ===
    print("[2/6] Teknik İndikatörler Hazırlanıyor...")

    # Teknik Featurelar
    processed_df = build_features.add_commodity_features(processed_df, date_col="Date")
    processed_df = build_features.add_technical_indicators(processed_df)
    processed_df = build_features.create_lagged_features(processed_df, 'Index_Change_Percent')
    processed_df = build_features.add_stock_mention_feature(processed_df, out_col="mentions_stock")
    processed_df = build_features.add_company_mention_feature(processed_df)

    # Temizlik
    processed_df = processed_df.dropna().reset_index(drop=True)

    # Hedef Değişken
    processed_df['Target_Index_Change'] = processed_df['Index_Change_Percent'].shift(-1)
    processed_df = processed_df.dropna(subset=['Target_Index_Change'])

    # Threshold Filtreleme
    THRESHOLD = 0.5
    processed_df = processed_df[abs(processed_df['Target_Index_Change']) > THRESHOLD]
    processed_df['Target_Direction'] = (processed_df['Target_Index_Change'] > 0).astype(int)

    print(f"Eğitime girecek satır sayısı: {len(processed_df)}")

    # === ADIM 3: FinBERT Embedding ===
    print("[3/6] Embeddingler hesaplanıyor...")
    extractor = build_features.FinbertFeatureExtractor()

    headlines = processed_df['Headline'].tolist()
    embeddings = []
    sentiments = []

    total = len(headlines)
    for i, text in enumerate(headlines):
        if i % 200 == 0: print(f"  Embedding: {i}/{total}")
        embeddings.append(extractor.get_embedding(text))
        sentiments.append(extractor.get_sentiment(text))

    embeddings = np.array(embeddings)  # Shape: (N, 768)
    processed_df['sentiment'] = sentiments

    # === ADIM 4: Veri Bölme (Split) ===
    print("[4/6] Train/Val Bölümlemesi...")

    # Numeric Features Hazırlığı
    sentiment_dummies = pd.get_dummies(processed_df['sentiment'], prefix='sent')
    processed_df = pd.concat([processed_df, sentiment_dummies], axis=1)

    numeric_cols = [
        'Trading_Volume', 'Previous_Index_Change_Percent_1d', 'Volatility_7d',
        'RSI_14', 'Cumulative_Return_3d',
        'Gold_close_t1', 'Gold_ret1_t1', 'Oil_close_t1', 'Oil_ret1_t1',
        'keyword_bullish', 'keyword_bearish', 'mentions_stock',
        'sent_negative', 'sent_neutral', 'sent_positive'
    ]

    for col in numeric_cols:
        if col not in processed_df.columns: processed_df[col] = 0

    X_numeric = processed_df[numeric_cols].values
    y = processed_df['Target_Direction'].values

    # Kronolojik Split
    split_idx = int(len(processed_df) * 0.8)

    X_num_train = X_numeric[:split_idx]
    X_emb_train = embeddings[:split_idx]
    y_train = y[:split_idx]

    X_num_val = X_numeric[split_idx:]
    X_emb_val = embeddings[split_idx:]
    y_val = y[split_idx:]

    # === ADIM 5: Scaling ve PCA (Dimension Reduction) ===
    print("[5/6] PCA ile Boyut İndirgeme (768 -> 50)...")

    # 1. Scaler (Sadece Numeric verilere)
    scaler = StandardScaler()
    X_num_train_scaled = scaler.fit_transform(X_num_train)
    X_num_val_scaled = scaler.transform(X_num_val)

    # 2. PCA (Sadece Embeddinglere)
    # 3000 veri için 50 component genelde %90+ varyansı tutar ve overfit engeller.
    N_COMPONENTS = 50
    pca = PCA(n_components=N_COMPONENTS, random_state=42)

    # PCA'yi sadece Train setine fit ediyoruz (Data leakage önlemek için)
    X_emb_train_pca = pca.fit_transform(X_emb_train)
    X_emb_val_pca = pca.transform(X_emb_val)

    print(f"  Varyansın ne kadarı korundu?: %{np.sum(pca.explained_variance_ratio_) * 100:.2f}")

    # 3. Birleştirme (Numeric + PCA Embeddings)
    X_train_final = np.concatenate([X_num_train_scaled, X_emb_train_pca], axis=1)
    X_val_final = np.concatenate([X_num_val_scaled, X_emb_val_pca], axis=1)

    print(f"  Eğitim Matrisi Son Boyutu: {X_train_final.shape}")
    # Beklenen: (Satır, 15 + 50 = 65)

    # === ADIM 6: Model Eğitimi ve Kayıt ===
    print("[6/6] Model Eğitiliyor...")

    rf_model = train_model.train_rf_optimized(
        X_train=X_train_final,
        y_train=y_train,
        X_val=X_val_final,
        y_val=y_val
    )

    if not os.path.exists('models'):
        os.makedirs('models')

    joblib.dump(rf_model, 'models/rf_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(pca, 'models/pca.pkl')  # <--- PCA'Yİ KAYDETMEYİ UNUTMUYORUZ

    print("\nBAŞARILI! Model ve PCA kaydedildi.")
    print("models/rf_model.pkl, models/scaler.pkl, models/pca.pkl")


if __name__ == "__main__":
    main()