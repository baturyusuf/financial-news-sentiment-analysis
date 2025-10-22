import pandas as pd  # Gerekli importları ekleyin
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
import torch


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