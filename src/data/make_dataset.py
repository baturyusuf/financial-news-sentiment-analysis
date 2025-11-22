import pandas as pd
from sklearn.model_selection import train_test_split  # Bu import'u ekleyin


def load_raw_data(path: str) -> pd.DataFrame:
    """Ham veriyi yükler."""
    # Notebook'taki veri yükleme satırınız
    return pd.read_csv(path)


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Eksik verileri yönetir ve temel temizliği yapar."""
    # Notebook'taki .fillna() satırlarınız [cite: 207]
    df['Index_Change_Percent'] = df['Index_Change_Percent'].fillna(df['Index_Change_Percent'].median())
    df['Trading_Volume'] = df['Trading_Volume'].fillna(df['Trading_Volume'].median())

    # Notebook'taki .dropna() satırınız [cite: 208]
    df.dropna(subset=['Headline'], inplace=True)
    return df


def create_target_variables(df: pd.DataFrame, threshold=0.005) -> pd.DataFrame:
    """
    Sadece %0.5'ten büyük değişimleri hedef al.
    Aradaki küçük değişimleri ya 'Nötr' yap ya da eğitimden çıkar.
    """
    # Örnek: Küçük değişimleri veri setinden atarak modeli net sinyallere odakla
    mask = abs(df['Index_Change_Percent']) > threshold
    df_filtered = df[mask].copy()

    df_filtered['Movement_Direction'] = (df_filtered['Index_Change_Percent'] > 0).astype(int)
    return df_filtered


def split_data_chronological(df: pd.DataFrame, test_size=0.15, val_size=0.15):
    """Veriyi sızdırmadan, zamana göre böler."""
    # Notebook'taki veri bölme mantığınız [cite: 271-272]
    val_split_idx = int(len(df) * (1 - test_size - val_size))
    test_split_idx = int(len(df) * (1 - test_size))

    train_df = df.iloc[:val_split_idx]
    val_df = df.iloc[val_split_idx:test_split_idx]
    test_df = df.iloc[test_split_idx:]

    return train_df, val_df, test_df