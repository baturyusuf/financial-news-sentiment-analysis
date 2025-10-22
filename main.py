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
    processed_df = build_features.create_lagged_features(processed_df, 'Index_Change_Percent')

    # === ADIM 4: Hibrit Vektör ve Veri Bölme ===
    print("Veri, eğitim ve test setlerine bölünüyor...")

    # Sayısal özellikleri ve duygu skorunu hazırla
    sentiment_dummies = pd.get_dummies(processed_df['sentiment'], prefix='sent')
    numerical_features = processed_df[['Trading_Volume', 'Previous_Index_Change_Percent_1d']]
    numerical_features = pd.concat([numerical_features, sentiment_dummies], axis=1)

    # Özellikleri ölçeklendir (XGBoost için şart değil ama LSTM için önemli)
    scaler = StandardScaler()
    numerical_features_scaled = scaler.fit_transform(numerical_features)

    # Sayısal özellikleri ve FinBERT gömmelerini birleştir
    # --- TEST: 768 boyutlu vektörleri geçici olarak devre dışı bırak ---
    X_hybrid = numerical_features_scaled
    # X_hybrid = np.concatenate([numerical_features_scaled, embeddings], axis=1)

    # Hedef değişkenleri al
    y_classification = processed_df['Movement_Direction'].values
    y_regression = processed_df['Index_Change_Percent'].values

    # Veriyi zamansal olarak böl (Indeksleri kullanarak)
    train_df, val_df, test_df = make_dataset.split_data_chronological(processed_df)

    train_indices = train_df.index
    val_indices = val_df.index

    # Sınıflandırma Verisi (XGBoost için)
    X_train_clf = X_hybrid[train_indices]
    y_train_clf = y_classification[train_indices]
    X_val_clf = X_hybrid[val_indices]
    y_val_clf = y_classification[val_indices]

    # Regresyon Verisi (LSTM için)
    # Not: LSTM için X_hybrid ve y_regression verilerini sıralı hale getirmeliyiz

    # === ADIM 5: Model Eğitimi ===
    print("Modeller eğitiliyor...")

    # Model 1: XGBoost Sınıflandırıcı
    train_model.train_xgboost_classifier(X_train_clf, y_train_clf, X_val_clf, y_val_clf)

    # Model 2 & 3: XGBoost ve LSTM Regresörler...
    # (LSTM için 'train_model.create_lstm_sequences' fonksiyonunu burada çağırın)
    # ...

    print("Pipeline tamamlandı.")


if __name__ == "__main__":
    main()