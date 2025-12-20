import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.data import make_dataset
from src.features import build_features


def diagnose():
    print("=== TANI TESTİ BAŞLIYOR ===")

    # 1. Veriyi Yükle ve İşle (Main.py ile aynı adımlar)
    raw_df = make_dataset.load_raw_data('data/raw/financial_news_market_events_2025.csv')
    df = make_dataset.preprocess_data(raw_df)
    df = make_dataset.create_target_variables(df)
    df = df.reset_index(drop=True)

    # Feature'ları ekle
    df = build_features.add_commodity_features(df, date_col="Date", debug=False)
    df = build_features.add_technical_indicators(df)

    # Regex Kontrollerini Çalıştır
    print("\n[TEST 1] Regex / Şirket Eşleşme Kontrolü...")
    df = build_features.add_company_mention_feature(df, out_col="mentions_company")
    df = build_features.add_stock_mention_feature(df, out_col="mentions_stock")

    count_comp = df['mentions_company'].sum()
    count_stock = df['mentions_stock'].sum()
    print(f"Toplam Satır: {len(df)}")
    print(f"Şirket Eşleşmesi Bulunan Satır: {count_comp} (%{count_comp / len(df) * 100:.2f})")
    print(f"Ticker Eşleşmesi Bulunan Satır : {count_stock} (%{count_stock / len(df) * 100:.2f})")

    if count_comp == 0 and count_stock == 0:
        print("KRİTİK HATA: Regex hiç eşleşme bulamıyor! build_features.py mantığı incelenmeli.")

    # Target Shift İşlemi
    df['Target_Direction'] = df['Movement_Direction'].shift(-1)
    df = df.dropna(subset=['Target_Direction'])

    # 2. Sınıf Dengesizliği (Class Imbalance)
    print("\n[TEST 2] Sınıf Dağılımı (Target Balance)...")
    balance = df['Target_Direction'].value_counts(normalize=True)
    print(balance)

    if abs(balance[0] - balance[1]) > 0.1:  # %10'dan fazla fark varsa
        print("UYARI: Veri seti dengesiz! Model çoğunluk sınıfına bias geliştiriyor.")

    # 3. Teknik İndikatör Korelasyonu
    print("\n[TEST 3] Teknik İndikatörlerin Hedefle İlişkisi...")
    corr_matrix = df[['Target_Direction', 'SMA_7', 'Momentum', 'Index_Change_Percent']].corr()
    print(corr_matrix['Target_Direction'])

    if abs(corr_matrix.loc['SMA_7', 'Target_Direction']) < 0.01:
        print("UYARI: SMA_7'nin hedefle ilişkisi neredeyse sıfır. Hesaplama hatası veya piyasa çok gürültülü.")

    print("\n=== TANI TESTİ BİTTİ ===")


if __name__ == "__main__":
    diagnose()