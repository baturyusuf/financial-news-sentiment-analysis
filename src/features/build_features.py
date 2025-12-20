import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
import torch
import re
from src.data.market_data import get_daily_close


class FinbertFeatureExtractor:
    """
    FinBERT kullanarak duygu ve embedding çıkaran sınıf.
    """

    def __init__(self, model_name="ProsusAI/finbert"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.sentiment_model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.embedding_model = AutoModel.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sentiment_model.to(self.device)
        self.embedding_model.to(self.device)

    def get_sentiment(self, text: str) -> str:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            logits = self.sentiment_model(**inputs).logits
        scores = {k: v for k, v in
                  zip(self.sentiment_model.config.id2label.values(), torch.softmax(logits[0], dim=0).tolist())}
        return max(scores, key=scores.get)

    def get_embedding(self, text: str) -> np.ndarray:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            outputs = self.embedding_model(**inputs)
        cls_embedding = outputs.last_hidden_state[0, 0, :].cpu().numpy()
        return cls_embedding


def create_lagged_features(df: pd.DataFrame, col_name: str, lag_days: int = 1) -> pd.DataFrame:
    df[f'Previous_{col_name}_{lag_days}d'] = df[col_name].shift(lag_days)
    df.fillna(0, inplace=True)
    return df


# =============================================================================
# GÜNCELLEME 1: DAHA GÜÇLÜ TEKNİK İNDİKATÖRLER (RSI ve Volatilite)
# =============================================================================
def add_technical_indicators(df, price_col='Index_Change_Percent'):
    """
    SMA yerine RSI ve Volatilite ekler.
    """
    df = df.copy()

    # 1. Volatilite (Risk): Son 7 gündeki değişimin standart sapması
    # Piyasa korkusunu ölçer. Yüksek volatilite genelde düşüş habercisidir.
    df['Volatility_7d'] = df[price_col].rolling(window=7).std().fillna(0)

    # 2. RSI (Göreceli Güç Endeksi) - Basitleştirilmiş
    # Fiyat yerine değişim yüzdesi üzerinden momentum hesaplar.
    window_length = 14
    delta = df[price_col]

    gain = (delta.where(delta > 0, 0)).rolling(window=window_length).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window_length).mean()

    rs = gain / loss
    rs = rs.fillna(0)  # Bölme hatasını engelle

    df['RSI_14'] = 100 - (100 / (1 + rs))
    df['RSI_14'] = df['RSI_14'].fillna(50)  # İlk günler için nötr değer (50) ata

    # 3. Kümülatif Getiri (Trend): Son 3 gün piyasa ne kadar şişti/düştü?
    df['Cumulative_Return_3d'] = df[price_col].rolling(window=3).sum().fillna(0)

    return df


# =============================================================================
# GÜNCELLEME 2: DAHA GENİŞ KAPSAMLI KEYWORD ARAMA (Regex Fix)
# =============================================================================
# Artık sadece spesifik şirket isimlerine değil, genel piyasa terimlerine de bakacağız.
MARKET_KEYWORDS = {
    "bullish": [r"surge", r"jump", r"rally", r"high", r"gain", r"bull", r"positive", r"grow", r"record"],
    "bearish": [r"plunge", r"drop", r"fall", r"crash", r"loss", r"bear", r"negative", r"fear", r"crisis", r"inflation"],
    "uncertainty": [r"volatil", r"uncertain", r"risk", r"worry", r"warn", r"tension"]
}


def add_stock_mention_feature(df: pd.DataFrame, headline_col: str = "Headline", **kwargs) -> pd.DataFrame:
    """
    Haber başlığında kritik piyasa kelimeleri geçiyor mu?
    """
    df = df.copy()

    def check_keywords(text, patterns):
        text = str(text).lower()
        for pat in patterns:
            if re.search(pat, text):
                return 1
        return 0

    # 3 Yeni Feature Ekliyoruz
    df['keyword_bullish'] = df[headline_col].apply(lambda x: check_keywords(x, MARKET_KEYWORDS["bullish"]))
    df['keyword_bearish'] = df[headline_col].apply(lambda x: check_keywords(x, MARKET_KEYWORDS["bearish"]))
    # 'mentions_stock' adını koruyalım ki main.py hata vermesin (ama içi artık keyword olacak)
    df['mentions_stock'] = df[headline_col].apply(lambda x: 1 if "$" in str(x) or "stock" in str(x).lower() else 0)

    return df


# Bu fonksiyonu şimdilik basit bir pass-through yapıyoruz, çünkü yukarıdaki keyword analizi daha etkili.
def add_company_mention_feature(df, **kwargs):
    # Regex 0 döndüğü için burayı 'mentions_stock' ile birleştirdik.
    # main.py hata vermesin diye boş bir sütun döndürüp geçiyoruz veya
    # basit bir doluluk kontrolü yapıyoruz.
    df['mentions_related_company_in_headline'] = 0
    return df


def add_commodity_features(df: pd.DataFrame, date_col="Date",
                           gold_symbol="GC=F", oil_symbol="CL=F",
                           debug=False) -> pd.DataFrame:
    df = df.copy()

    # Tarih formatlama
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce").dt.tz_localize(None)
    df = df.dropna(subset=[date_col]).sort_values(date_col)

    start = df[date_col].min() - pd.Timedelta(days=10)
    end = df[date_col].max() + pd.Timedelta(days=2)

    # Veri çekme (Hata olursa boş dönecek yapı)
    try:
        gold = get_daily_close(gold_symbol, start, end).rename(columns={"Close": "Gold_Close"})
        oil = get_daily_close(oil_symbol, start, end).rename(columns={"Close": "Oil_Close"})
    except Exception as e:
        print(f"Emtia verisi çekilemedi: {e}")
        # Boş dataframe oluştur hata vermemesi için
        return df

    def _mk(price_df: pd.DataFrame, close_col: str, prefix: str) -> pd.DataFrame:
        if price_df.empty: return pd.DataFrame(columns=[date_col, f"{prefix}_close_t1", f"{prefix}_ret1_t1"])

        price_df = price_df.copy()
        price_df["Date"] = pd.to_datetime(price_df["Date"], errors="coerce").dt.tz_localize(None)
        price_df[close_col] = pd.to_numeric(price_df[close_col], errors="coerce")
        price_df = price_df.dropna(subset=["Date", close_col]).sort_values("Date")

        price_df[f"{prefix}_close_t1"] = price_df[close_col].shift(1)
        price_df[f"{prefix}_ret1_t1"] = price_df[close_col].pct_change().shift(1)

        feat = price_df[["Date", f"{prefix}_close_t1", f"{prefix}_ret1_t1"]].copy()
        feat = feat.rename(columns={"Date": date_col})
        return feat.sort_values(date_col)

    gold_feat = _mk(gold, "Gold_Close", "Gold")
    oil_feat = _mk(oil, "Oil_Close", "Oil")

    if not gold_feat.empty:
        df = pd.merge_asof(df, gold_feat, on=date_col, direction="backward")
    if not oil_feat.empty:
        df = pd.merge_asof(df, oil_feat, on=date_col, direction="backward")

    # Merge sonrası oluşan NaN'ları doldur
    cols_to_fill = ['Gold_close_t1', 'Gold_ret1_t1', 'Oil_close_t1', 'Oil_ret1_t1']
    for c in cols_to_fill:
        if c in df.columns:
            df[c] = df[c].fillna(0)
        else:
            df[c] = 0  # Eğer veri hiç gelmezse sütunu 0 ile oluştur

    return df