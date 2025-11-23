import nltk
nltk.download('wordnet')
nltk.download('stopwords')

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# 1. Veri Yükleme ve İşleme Modüllerimiz
from src.data import make_dataset
# 2. Özellik Mühendisliği Modüllerimiz
from src.features import text_preprocessing, build_features
# 3. Model Eğitme Modüllerimiz
from src.models import train_model


def main():
    print("Proje pipeline'ı başlıyor...")

    # === ADIM 1: Veri Yükleme ve Ön İşleme ===
    print("Veri yükleniyor ve ön işleniyor...")
    raw_df = make_dataset.load_raw_data('data/raw/financial_news_market_events_2025.csv')
    processed_df = make_dataset.preprocess_data(raw_df)
    processed_df = make_dataset.create_target_variables(processed_df)

    # === ADIM 2: Metin İşleme ===
    print("Metin verisi temizleniyor (NLP)...")
    # .apply() notebook'ta yavaşsa burada pandas_apply kullanabilirsiniz (tqdm ile)
    # Metin temizlemeyi SADECE eski modeller için saklayın, BERT için DEĞİL.
    # processed_df['cleaned_headline'] = processed_df['Headline'].apply(text_preprocessing.full_text_pipeline)
    ...
    extractor = build_features.FinbertFeatureExtractor()
    # FinBERT'e ORİJİNAL, ham başlığı verin
    processed_df['sentiment'] = processed_df['Headline'].apply(extractor.get_sentiment)
    embeddings = np.array([extractor.get_embedding(text) for text in processed_df['Headline']])

    # Diğer özellikleri ekle
    processed_df = build_features.add_technical_indicators(processed_df)
    processed_df = build_features.create_lagged_features(processed_df, 'Index_Change_Percent')

    # === ADIM 4: Hibrit Vektör ve Veri Bölme ===
    print("Veri, eğitim ve test setlerine bölünüyor...")

    # 1. Önce DataFrame'i zamana göre bölüyoruz (Burası ÇOK ÖNEMLİ: Scaler'dan önce yapılmalı)
    train_df, val_df, test_df = make_dataset.split_data_chronological(processed_df)

    # İndeksleri alalım (Numpy array'lerden veriyi çekmek için lazım olacak)
    train_idx = train_df.index
    val_idx = val_df.index
    # test_idx = test_df.index # İleride test için lazım olursa

    # 2. Kullanılacak Sayısal Özellikleri Belirle
    # Not: 'SMA_7' ve 'Momentum' build_features'ta eklediğimiz yeni sütunlar
    numeric_cols = ['Trading_Volume', 'Previous_Index_Change_Percent_1d', 'SMA_7', 'Momentum']

    # Eğer sentiment dummy kullanıyorsan onları da listeye ekle veya ayrı tut
    sentiment_dummies = pd.get_dummies(processed_df['sentiment'], prefix='sent')

    # Tüm sayısal veriyi geçici olarak birleştir (Henüz scale etme!)
    X_numeric_all = pd.concat([processed_df[numeric_cols], sentiment_dummies], axis=1)

    # 3. Scaler'ı SADECE Eğitim (Train) verisi üzerinde eğit (.fit)
    scaler = StandardScaler()

    # Sadece train indekslerine denk gelen veriyi alıp fit ediyoruz
    scaler.fit(X_numeric_all.loc[train_idx])

    # 4. Şimdi tüm veriyi dönüştür (.transform)
    X_numeric_scaled = scaler.transform(X_numeric_all)  # Bu bize numpy array döndürür

    # 5. Sayısal Veriler ile BERT Embeddinglerini Birleştir (Modelin Gözünü Açıyoruz!)
    # X_numeric_scaled (Sayısal) + embeddings (Metin)
    X_hybrid = np.concatenate([X_numeric_scaled, embeddings], axis=1)

    # 6. Son olarak X ve y verilerini train/val olarak ayır

    # Hedef değişkenler
    y_classification = processed_df['Movement_Direction'].values
    y_regression = processed_df['Index_Change_Percent'].values # LSTM için gerekirse

    # Sınıflandırma Verisi (XGBoost için)
    X_train_clf = X_hybrid[train_idx]
    y_train_clf = y_classification[train_idx]

    X_val_clf = X_hybrid[val_idx]
    y_val_clf = y_classification[val_idx]

    # Regresyon Verisi (LSTM için)
    # Not: LSTM için X_hybrid ve y_regression verilerini sıralı hale getirmeliyiz

    # === ADIM 5: Model Eğitimi ===
    print("Modeller eğitiliyor...")

    # Sayısal özelliklerin isimlerini alalım
    numeric_feature_names = list(X_numeric_all.columns)

    # FinBERT embeddingleri için yapay isimler oluştur (embed_0, embed_1...)
    # Embedding boyutu 768
    embedding_feature_names = [f'bert_emb_{i}' for i in range(embeddings.shape[1])]

    # Tüm özellik isimlerini birleştir
    all_feature_names = numeric_feature_names + embedding_feature_names

    # Model 1: XGBoost Sınıflandırıcı (Feature Names parametresini ekledik)
    train_model.train_xgboost_classifier(X_train_clf, y_train_clf, X_val_clf, y_val_clf,
                                         feature_names=all_feature_names)

    # Model 2: LSTM (Eğer aktif edeceksen)
    # ... (XGBoost kodları bittikten hemen sonra buraya yapıştır) ...

    print("\n" + "=" * 30)
    print("LSTM EĞİTİM SÜRECİ BAŞLIYOR")
    print("=" * 30)

    # 1. Parametreler
    SEQ_LENGTH = 5  # Model geçmiş 5 güne bakarak tahmin yapacak

    # 2. Veriyi LSTM formatına (Samples, Timesteps, Features) dönüştür
    # X_hybrid: Hibrit özellikler (Sayısal + FinBERT)
    # y_regression: Hedef değişken (Index_Change_Percent)
    print(f"LSTM dizileri oluşturuluyor (Geçmiş {SEQ_LENGTH} gün)...")
    X_seq, y_seq = train_model.create_lstm_sequences(X_hybrid, y_regression, SEQ_LENGTH)

    # 3. Sıralı Bölme (Chronological Split)
    # Dizileme işlemi ilk 5 satırı yuttuğu için indeksleri yeniden hesaplıyoruz
    total_len = len(X_seq)
    train_size = int(total_len * 0.70)
    val_size = int(total_len * 0.15)

    # Train Seti
    X_train_lstm = X_seq[:train_size]
    y_train_lstm = y_seq[:train_size]

    # Validation Seti
    X_val_lstm = X_seq[train_size: train_size + val_size]
    y_val_lstm = y_seq[train_size: train_size + val_size]

    # (Opsiyonel) Test Seti - Kalan kısım
    # X_test_lstm = X_seq[train_size + val_size :]
    # y_test_lstm = y_seq[train_size + val_size :]

    print(f"LSTM Veri Boyutları -> Train: {X_train_lstm.shape}, Val: {X_val_lstm.shape}")

    # 4. Modelin Beklediği Input Şekli (Timesteps, Features)
    # Örnek: (5, 772) -> 5 gün, 772 özellik (768 BERT + 4 Sayısal)
    input_shape = (X_train_lstm.shape[1], X_train_lstm.shape[2])

    # 5. Modeli Eğit
    train_model.train_lstm_regressor(
        X_train_lstm,
        y_train_lstm,
        X_val_lstm,
        y_val_lstm,
        input_shape
    )

    print("Pipeline tamamlandı.")


if __name__ == "__main__":
    main()