# Dosya: src/utils_live.py
import yfinance as yf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import requests
import xml.etree.ElementTree as ET
import re


# === LSTM MODELİ ===
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=50, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq.view(len(input_seq), 1, -1))
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]


def get_stock_data(ticker, period="2y"):
    """Yahoo Finance'ten hisse verisi çeker."""
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
        return df, stock
    except Exception as e:
        print(f"Hisse verisi çekme hatası: {e}")
        return pd.DataFrame(), None


def train_and_predict_lstm(df):
    """LSTM Eğitimi ve Tahmini."""
    if df.empty: return 0.0

    data = df['Close'].values.astype(float)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    data_normalized = scaler.fit_transform(data.reshape(-1, 1))
    data_normalized = torch.FloatTensor(data_normalized).view(-1)

    train_window = 14

    if len(data_normalized) <= train_window:
        return data[-1]  # Yeterli veri yoksa son fiyatı dön

    def create_inout_sequences(input_data, tw):
        inout_seq = []
        L = len(input_data)
        for i in range(L - tw):
            train_seq = input_data[i:i + tw]
            train_label = input_data[i + tw:i + tw + 1]
            inout_seq.append((train_seq, train_label))
        return inout_seq

    train_inout_seq = create_inout_sequences(data_normalized, train_window)
    model = LSTMModel()
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.train()
    epochs = 15  # Hız için
    for i in range(epochs):
        for seq, labels in train_inout_seq:
            optimizer.zero_grad()
            y_pred = model(seq)
            single_loss = loss_function(y_pred, labels)
            single_loss.backward()
            optimizer.step()

    model.eval()
    last_seq = data_normalized[-train_window:].tolist()
    with torch.no_grad():
        pred_normalized = model(torch.FloatTensor(last_seq)).item()

    predicted_price = scaler.inverse_transform(np.array([[pred_normalized]]))
    return predicted_price[0][0]


# === GÜNCELLENMİŞ HABER ÇEKME FONKSİYONU (FALLBACKLİ) ===
def get_news_for_ticker(ticker_obj):
    """
    Yahoo Finance'i dener, çalışmazsa Google News RSS kullanır.
    """
    headlines = []
    ticker_symbol = ticker_obj.ticker if hasattr(ticker_obj, 'ticker') else "STOCK"

    # YÖNTEM 1: Yahoo Finance (yfinance)
    try:
        print(f"Yahoo News deneniyor: {ticker_symbol}...")
        news_list = ticker_obj.news
        if news_list:
            for news in news_list:
                title = news.get('title', news.get('headline', ''))
                link = news.get('link', news.get('url', '#'))
                publisher = news.get('publisher', 'Yahoo Finance')

                if title:
                    headlines.append({'title': title, 'link': link, 'publisher': publisher})
    except Exception as e:
        print(f"Yahoo News hatası: {e}")

    # Eğer Yahoo'dan haber gelmediyse -> YÖNTEM 2: Google News RSS
    if not headlines:
        print("Yahoo'dan haber gelmedi, Google News RSS deneniyor...")
        try:
            # Google News RSS URL'i (İngilizce - FinBERT için)
            rss_url = f"https://news.google.com/rss/search?q={ticker_symbol}+stock+news&hl=en-US&gl=US&ceid=US:en"
            response = requests.get(rss_url, timeout=5)

            if response.status_code == 200:
                root = ET.fromstring(response.content)
                # İlk 5 haberi al
                for item in root.findall('./channel/item')[:5]:
                    title = item.find('title').text if item.find('title') is not None else "No Title"
                    link = item.find('link').text if item.find('link') is not None else "#"
                    source = item.find('source').text if item.find('source') is not None else "Google News"

                    # Google başlıklarında genelde " - Publisher" eki olur, temizleyelim (opsiyonel)
                    clean_title = title

                    headlines.append({'title': clean_title, 'link': link, 'publisher': source})
        except Exception as e:
            print(f"Google News RSS hatası: {e}")

    return headlines