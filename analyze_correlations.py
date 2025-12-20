import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.data import make_dataset
from src.features import build_features


def analyze():
    print("=== KORELASYON ANALİZİ BAŞLIYOR ===")

    # 1. Veri Yükle
    raw_df = make_dataset.load_raw_data('data/raw/financial_news_market_events_2025.csv')
    df = make_dataset.preprocess_data(raw_df)

    # 2. İndikatörleri Ekle
    df = build_features.add_technical_indicators(df)
    df = build_features.add_commodity_features(df, date_col="Date")

    # 3. Sentiment Skorunu Sayısallaştır
    extractor = build_features.FinbertFeatureExtractor()
    print("Sentiment hesaplanıyor (Örneklem üzerinde)...")
    # Hız için son 500 satırı alalım yeter
    df_sample = df.tail(1000).copy()

    df_sample['sentiment_label'] = df_sample['Headline'].apply(extractor.get_sentiment)
    sentiment_map = {'positive': 1, 'neutral': 0, 'negative': -1}
    df_sample['Sentiment_Score'] = df_sample['sentiment_label'].map(sentiment_map)

    # 4. HEDEF: Yarının Mutlak Değişimi (Volatilite) ve Yönü
    df_sample['Target_Return'] = df_sample['Index_Change_Percent'].shift(-1)
    df_sample['Target_Volatility'] = abs(df_sample['Target_Return'])  # Ne kadar oynayacak?

    df_sample = df_sample.dropna()

    # 5. Korelasyon Matrisi
    cols_to_check = [
        'Target_Return',
        'Target_Volatility',
        'Sentiment_Score',
        'RSI_14',
        'Volatility_7d',
        'Previous_Index_Change_Percent_1d',
        'Trading_Volume'
    ]

    # Sütunların varlığını kontrol et
    valid_cols = [c for c in cols_to_check if c in df_sample.columns]

    corr_matrix = df_sample[valid_cols].corr()

    print("\n--- KORELASYON TABLOSU ---")
    print(corr_matrix[['Target_Return', 'Target_Volatility']])

    # 6. Görselleştirme
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Feature vs Target Korelasyonu")
    plt.tight_layout()
    plt.savefig('results/correlation_analysis.png')
    print("\nGrafik 'results/correlation_analysis.png' konumuna kaydedildi.")

    # YORUM
    sent_corr = corr_matrix.loc['Sentiment_Score', 'Target_Return']
    print(f"\nSentiment ile Fiyat Yönü İlişkisi: {sent_corr:.4f}")

    if abs(sent_corr) < 0.05:
        print("SONUÇ: Sentiment ile Fiyat Yönü arasında İLİŞKİ YOK.")
        print("Tavsiye: Yön tahmini yerine Volatilite tahminine geçilmeli.")


if __name__ == "__main__":
    analyze()