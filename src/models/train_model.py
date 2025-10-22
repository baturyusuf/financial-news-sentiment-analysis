# Gerekli importları ekleyin

from xgboost import XGBClassifier, XGBRegressor
from xgboost.callback import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import accuracy_score, mean_squared_error, f1_score
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np
import os


def create_lstm_sequences(data: np.ndarray, target: np.ndarray, seq_length: int):
    """LSTM için (samples, timesteps, features) formatında veri hazırlar."""
    # Notebook'taki LSTM veri hazırlama döngünüz
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length)])
        y.append(target[i + seq_length])
    return np.array(X), np.array(y)


def train_xgboost_classifier(X_train, y_train, X_val, y_val):
    """XGBoost Sınıflandırıcıyı eğitir ve değerlendirir."""

    # 'use_label_encoder=False' parametresi bu eski sürüm için gereklidir.
    # 'eval_metric='logloss'' da ekleyelim.
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', early_stopping_rounds=10)

    # 'callbacks' yerine 'early_stopping_rounds' parametresini KULLANIN
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              verbose=False)

    # Değerlendirme
    preds = model.predict(X_val)
    acc = accuracy_score(y_val, preds)
    f1 = f1_score(y_val, preds, average='weighted')
    print(f"[XGB Classifier] Accuracy: {acc:.4f}, F1-Score: {f1:.4f}")

    # Kaydetme
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/xgb_classifier.joblib')
    return model


# ... Benzer şekilde train_xgboost_regressor fonksiyonu ... [cite: 254]

def train_lstm_regressor(X_train_seq, y_train_seq, X_val_seq, y_val_seq, input_shape):
    """LSTM Regresör modelini eğitir ve değerlendirir."""
    # Notebook'taki LSTM model tanımınız [cite: 257-259]
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(25))
    model.add(Dense(1))  # Regresyon için 1 nöron [cite: 212]

    model.compile(optimizer='adam', loss='mean_squared_error')

    # Notebook'taki .fit() çağrınız
    model.fit(X_train_seq, y_train_seq,
              validation_data=(X_val_seq, y_val_seq),
              batch_size=32,
              epochs=50,
              callbacks=[...])  # EarlyStopping callback'i ekleyebilirsiniz

    # Değerlendirme
    mse = model.evaluate(X_val_seq, y_val_seq, verbose=0)
    print(f"[LSTM Regressor] Validation MSE: {mse:.4f}")

    # Kaydetme
    os.makedirs('models', exist_ok=True)
    model.save('models/lstm_regressor.h5')
    return model