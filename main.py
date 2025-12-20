import nltk

nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Modüller
from src.data import make_dataset
from src.features import text_preprocessing, build_features
from src.models import train_model


def main():
    print("=" * 60)
    print("PROJE PIPELINE (CORRECT SEQUENCE VERSION)")
    print("=" * 60)

    # === ADIM 1: Veri Yükleme ===
    print("[1/6] Veri yükleniyor...")
    raw_df = make_dataset.load_raw_data('data/raw/financial_news_market_events_2025.csv')

    # İndeksi baştan güvenceye alalım
    processed_df = make_dataset.preprocess_data(raw_df)
    processed_df = processed_df.reset_index(drop=True)

    # === ADIM 2: ÖZELLİK MÜHENDİSLİĞİ (FİLTRELEMEDEN ÖNCE YAPILMALI!) ===
    # Zaman serisi bozulmadan önce indikatörleri hesaplıyoruz.
    print("[2/6] Özellikler ekleniyor (Zaman serisi bozulmadan)...")

    # 2.1 Emtia
    processed_df = build_features.add_commodity_features(processed_df, date_col="Date")

    # 2.2 Teknik İndikatörler (RSI, Volatilite)
    processed_df = build_features.add_technical_indicators(processed_df)

    # 2.3 Lag Özellikleri (Geçmiş günler)
    processed_df = build_features.create_lagged_features(processed_df, 'Index_Change_Percent')

    # 2.4 Keyword / Şirket Taraması
    processed_df = build_features.add_stock_mention_feature(processed_df, out_col="mentions_stock")
    processed_df = build_features.add_company_mention_feature(processed_df)

    # Lag işlemi (shift) yüzünden oluşan ilk satırlardaki NaN'ları temizle
    processed_df = processed_df.dropna()
    processed_df = processed_df.reset_index(drop=True)  # İndeksi tazeleyelim

    # === ADIM 3: HEDEF BELİRLEME VE FİLTRELEME ===
    # === ADIM 3: HEDEF BELİRLEME VE FİLTRELEME ===
    print("[3/6] Hedef değişkenler ve Filtreleme...")

    # Hedef: Yarının değişimi
    processed_df['Target_Index_Change'] = processed_df['Index_Change_Percent'].shift(-1)

    # NaN olan (son gün) satırını at
    processed_df = processed_df.dropna(subset=['Target_Index_Change'])

    # --- VERİ ANALİZİ ---
    print("Veri İstatistiği (Tüm Veri):")
    print(processed_df['Target_Index_Change'].describe())

    # [GÜNCELLEME] SABİT EŞİK DEĞERİ (THRESHOLD)
    # %0.5 (Yani 0.5) altındaki hareketleri gürültü sayıp atıyoruz.
    # %3.6 gibi uçuk değerlere gitmiyoruz.
    THRESHOLD = 0.5

    abs_change = abs(processed_df['Target_Index_Change'])

    # Filtreleme
    processed_df = processed_df[abs_change > THRESHOLD]

    # Yön Belirleme
    processed_df['Target_Direction'] = (processed_df['Target_Index_Change'] > 0).astype(int)

    print(f"Eşik Değeri ({THRESHOLD}) Uygulandı.")
    print(f"FİLTRE SONRASI Eğitime girecek satır sayısı: {len(processed_df)}")
    # === ADIM 4: FinBERT Embedding (SENKRONİZASYON GARANTİLİ) ===
    print("[4/6] FinBERT Embeddingleri hesaplanıyor...")

    # DİKKAT: Embedding'i SADECE kalan satırlar için hesaplıyoruz.
    extractor = build_features.FinbertFeatureExtractor()

    embeddings_list = []
    sentiments_list = []

    headlines = processed_df['Headline'].tolist()
    total = len(headlines)

    print(f"Toplam {total} başlık işleniyor...")

    for i, text in enumerate(headlines):
        if i % 100 == 0: print(f"  Processed {i}/{total}")
        emb = extractor.get_embedding(text)
        sent = extractor.get_sentiment(text)
        embeddings_list.append(emb)
        sentiments_list.append(sent)

    embeddings = np.array(embeddings_list)
    processed_df['sentiment'] = sentiments_list

    print(f"Embedding Shape: {embeddings.shape}")
    print(f"DataFrame Shape: {processed_df.shape}")

    assert len(embeddings) == len(processed_df), "HATA: Embedding ve DataFrame satır sayıları tutmuyor!"

    # === ADIM 5: Veri Hazırlığı ===
    print("[5/6] Veri setleri bölünüyor...")

    # Kronolojik Bölme
    train_df, val_df, test_df = make_dataset.split_data_chronological(processed_df)

    train_idx = range(0, len(train_df))
    val_idx = range(len(train_df), len(train_df) + len(val_df))

    # --- HİLE TESTİ (KAPALI) ---
    # processed_df['CHEAT_CODE'] = processed_df['Target_Index_Change']
    # ---------------------------

    numeric_cols = [
        # 'CHEAT_CODE',
        'Trading_Volume',
        'Previous_Index_Change_Percent_1d',
        'Volatility_7d',
        'RSI_14',
        'Cumulative_Return_3d',
        'Gold_close_t1', 'Gold_ret1_t1',
        'Oil_close_t1', 'Oil_ret1_t1',
        'keyword_bullish',
        'keyword_bearish',
        'mentions_stock'
    ]

    # Sentiment Dummies
    sentiment_dummies = pd.get_dummies(processed_df['sentiment'], prefix='sent')
    processed_df = pd.concat([processed_df, sentiment_dummies], axis=1)

    actual_numeric_cols = [c for c in numeric_cols if c in processed_df.columns]
    actual_numeric_cols += list(sentiment_dummies.columns)

    X_numeric_all = processed_df[actual_numeric_cols]

    # Scaler
    scaler = StandardScaler()
    scaler.fit(X_numeric_all.iloc[train_idx])
    X_numeric_scaled = scaler.transform(X_numeric_all)

    # Eğitim ve Val Setlerini Ayır
    X_num_train = X_numeric_scaled[train_idx]
    X_emb_train = embeddings[train_idx]
    y_train = processed_df.iloc[train_idx]['Target_Direction'].values

    X_num_val = X_numeric_scaled[val_idx]
    X_emb_val = embeddings[val_idx]
    y_val = processed_df.iloc[val_idx]['Target_Direction'].values

    # === ADIM 6: Model Eğitimi ===
    print("[6/6] Model Eğitimi (Random Forest)...")
    print(f"Kullanılan özellikler: {actual_numeric_cols}")

    train_model.train_rf_optimized(
        X_numeric=X_num_train,
        embeddings=X_emb_train,
        y_train=y_train,
        X_numeric_val=X_num_val,
        embeddings_val=X_emb_val,
        y_val=y_val,
        feature_names=actual_numeric_cols
    )

    print("\nTAMAMLANDI.")


if __name__ == "__main__":
    main()