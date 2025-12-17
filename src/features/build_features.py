import pandas as pd  # Gerekli importları ekleyin
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
import torch
import re
from src.data.market_data import get_daily_close
from src.data.market_data import get_daily_close


class FinbertFeatureExtractor:
    """
    FinBERT kullanarak duygu ve embedding çıkaran sınıf.
    """

    def __init__(self, model_name="ProsusAI/finbert"):
        # Notebook'taki tokenizer ve model yükleme kodlarınız
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Duygu için Sınıflandırma Modeli [cite: 230]
        self.sentiment_model = AutoModelForSequenceClassification.from_pretrained(model_name)
        # Gömme (Embedding) için Temel Model [cite: 231]
        self.embedding_model = AutoModel.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sentiment_model.to(self.device)
        self.embedding_model.to(self.device)

    def get_sentiment(self, text: str) -> str:
        """Bir metnin duygu etiketini (positive, negative, neutral) döndürür."""
        # Notebook'taki duygu tahmini kodunuz
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            logits = self.sentiment_model(**inputs).logits

        scores = {k: v for k, v in
                  zip(self.sentiment_model.config.id2label.values(), torch.softmax(logits[0], dim=0).tolist())}
        return max(scores, key=scores.get)

    def get_embedding(self, text: str) -> np.ndarray:
        """Bir metnin 768 boyutlu [CLS] token embedding'ini döndürür."""
        # Notebook'taki embedding çıkarma kodunuz
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            outputs = self.embedding_model(**inputs)

        cls_embedding = outputs.last_hidden_state[0, 0, :].cpu().numpy()
        return cls_embedding


def create_lagged_features(df: pd.DataFrame, col_name: str, lag_days: int = 1) -> pd.DataFrame:
    """Gecikme özellikleri ekler, örn: Önceki günün değişimi."""
    # Notebook'taki .shift() kodunuz
    df[f'Previous_{col_name}_{lag_days}d'] = df[col_name].shift(lag_days)
    df.fillna(0, inplace=True)  # shift() sonrası oluşan NaN'ları doldur
    return df


def add_technical_indicators(df, price_col='Index_Change_Percent'):
    """
    Veri setine teknik analiz indikatörleri ekler.
    """
    # 1. Basit Hareketli Ortalama (SMA - 7 Günlük)
    # Piyasanın haftalık trendini gösterir.
    df['SMA_7'] = df[price_col].rolling(window=7).mean()

    # 2. Momentum (4 Günlük)
    # 4 gün öncesine göre değişim hızı.
    df['Momentum'] = df[price_col] - df[price_col].shift(4)

    # 3. Eksik verileri (ilk 7 gün NaN olacaktır) doldur
    df['SMA_7'] = df['SMA_7'].fillna(0)
    df['Momentum'] = df['Momentum'].fillna(0)

    return df

_TICKER_LIKE = [
    re.compile(r"\$[A-Z]{1,5}\b"),                         # $AAPL
    re.compile(r"\b(?:NASDAQ|NYSE|AMEX)\s*:\s*[A-Z]{1,5}\b", re.I),  # NASDAQ: AAPL
    re.compile(r"\([A-Z]{1,5}\)"),                         # (AAPL)
]

def add_stock_mention_feature(
    df: pd.DataFrame,
    headline_col: str = "Headline",
    ticker_col: str | None = None,     # ör: "Ticker" / "Symbol"
    company_col: str | None = None,    # ör: "Company"
    out_col: str = "mentions_stock",
) -> pd.DataFrame:
    """
    Haber başlığında:
    - (varsa) satırın kendi ticker'ı / şirket adı geçiyor mu?
    - yoksa genel olarak ticker-like bir ifade var mı?
    """
    df = df.copy()

    def _row_check(row) -> int:
        text = str(row.get(headline_col, "") or "")
        text_up = text.upper()
        text_low = text.lower()

        # 1) Satıra özel ticker kontrolü (en güçlü sinyal)
        if ticker_col and ticker_col in df.columns and pd.notna(row.get(ticker_col)):
            t = str(row[ticker_col]).upper().strip()
            if t:
                # $TSLA veya kelime sınırı TSLA
                if re.search(rf"(?:\${re.escape(t)}\b|\b{re.escape(t)}\b)", text_up):
                    return 1

        # 2) Satıra özel şirket adı kontrolü (daha zayıf ama faydalı)
        if company_col and company_col in df.columns and pd.notna(row.get(company_col)):
            name = str(row[company_col]).strip()
            if name and (name.lower() in text_low):
                return 1

        # 3) Fallback: herhangi bir ticker-like pattern var mı?
        for pat in _TICKER_LIKE:
            if pat.search(text):
                return 1

        return 0

    df[out_col] = df.apply(_row_check, axis=1).astype(int)
    return df


def add_commodity_features(df: pd.DataFrame, date_col="Date",
                           gold_symbol="GC=F", oil_symbol="CL=F",
                           debug=False) -> pd.DataFrame:
    df = df.copy()

    # Date kolonu garanti + sıralama garanti
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce").dt.tz_localize(None)
    df = df.dropna(subset=[date_col]).sort_values(date_col)

    start = df[date_col].min() - pd.Timedelta(days=10)
    end   = df[date_col].max() + pd.Timedelta(days=2)

    gold = get_daily_close(gold_symbol, start, end).rename(columns={"Close": "Gold_Close"})
    oil  = get_daily_close(oil_symbol,  start, end).rename(columns={"Close": "Oil_Close"})

    def _mk(price_df: pd.DataFrame, close_col: str, prefix: str) -> pd.DataFrame:
        price_df = price_df.copy()
        price_df["Date"] = pd.to_datetime(price_df["Date"], errors="coerce").dt.tz_localize(None)
        price_df[close_col] = pd.to_numeric(price_df[close_col], errors="coerce")
        price_df = price_df.dropna(subset=["Date", close_col]).sort_values("Date")

        price_df[f"{prefix}_close_t1"] = price_df[close_col].shift(1)
        price_df[f"{prefix}_ret1_t1"]  = price_df[close_col].pct_change().shift(1)

        feat = price_df[["Date", f"{prefix}_close_t1", f"{prefix}_ret1_t1"]].copy()
        feat = feat.rename(columns={"Date": date_col})   # ✅ anahtar aynı isimde
        return feat.sort_values(date_col)

    gold_feat = _mk(gold, "Gold_Close", "Gold")
    oil_feat  = _mk(oil,  "Oil_Close",  "Oil")

    if debug:
        print("GOLD dtypes:\n", gold.dtypes)
        print("GOLD head:\n", gold.head())
        print("gold_feat head:\n", gold_feat.head())
        print("OIL dtypes:\n", oil.dtypes)
        print("OIL head:\n", oil.head())
        print("oil_feat head:\n", oil_feat.head())

    # ✅ on=date_col kullan — artık Date düşürmeye gerek yok
    df = pd.merge_asof(df, gold_feat, on=date_col, direction="backward")
    df = pd.merge_asof(df, oil_feat,  on=date_col, direction="backward")

    return df


_CORP_SUFFIX = re.compile(r"\b(inc|inc\.|corp|corp\.|co|co\.|ltd|ltd\.|plc|llc|sa|ag|nv|holdings?)\b", re.I)

def add_company_mention_feature(df: pd.DataFrame,
                                headline_col="Headline",
                                related_company_col="Related_Company",
                                out_col="mentions_related_company_in_headline") -> pd.DataFrame:
    df = df.copy()

    def normalize_name(name: str) -> str:
        name = str(name)
        name = name.replace("&", "and")
        name = _CORP_SUFFIX.sub(" ", name)
        name = re.sub(r"[^a-zA-Z0-9\s]", " ", name)
        name = re.sub(r"\s+", " ", name).strip().lower()
        return name

    def row_flag(row) -> int:
        headline = str(row.get(headline_col, "") or "")
        h = re.sub(r"[^a-zA-Z0-9\s]", " ", headline)
        h = re.sub(r"\s+", " ", h).strip().lower()

        rc = row.get(related_company_col, None)
        if rc is None or (isinstance(rc, float) and pd.isna(rc)) or str(rc).strip() == "":
            return 0

        # Birden fazla şirket virgülle gelirse:
        companies = [c.strip() for c in str(rc).split(",") if c.strip()]

        for comp in companies:
            comp_n = normalize_name(comp)
            if not comp_n:
                continue

            # Çok kısa parçaları at (yanlış pozitifleri azaltır)
            tokens = [t for t in comp_n.split() if len(t) >= 3]
            if not tokens:
                continue

            # Basit ama etkili: şirket adının en az bir ana token’ı başlıkta geçsin
            if any(re.search(rf"\b{re.escape(t)}\b", h) for t in tokens):
                return 1

        return 0

    df[out_col] = df.apply(row_flag, axis=1).astype(int)
    return df
