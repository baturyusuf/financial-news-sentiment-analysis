# src/models/train_model.py

from xgboost import XGBClassifier, XGBRegressor
from xgboost import plot_importance
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, \
    mean_squared_error
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Grafikleri kaydetmek için klasör oluştur
os.makedirs('results', exist_ok=True)


def create_lstm_sequences(data: np.ndarray, target: np.ndarray, seq_length: int):
    """LSTM için (samples, timesteps, features) formatında veri hazırlar."""
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length)])
        y.append(target[i + seq_length])
    return np.array(X), np.array(y)


def train_xgboost_classifier(X_train, y_train, X_val, y_val, feature_names=None):
    """
    XGBoost Sınıflandırıcıyı eğitir, Table I metriklerini basar
    ve Confusion Matrix + Feature Importance grafiklerini çizer.
    """
    print("\n[XGBoost] Model eğitiliyor...")
    model = XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss',
        early_stopping_rounds=10,
        n_estimators=1000,
        learning_rate=0.05
    )

    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              verbose=False)

    # --- 1. TABLE I Verileri (Metrikler) ---
    preds = model.predict(X_val)

    acc = accuracy_score(y_val, preds)
    f1 = f1_score(y_val, preds, average='weighted')
    prec = precision_score(y_val, preds, average='weighted')
    rec = recall_score(y_val, preds, average='weighted')

    print("-" * 40)
    print("TABLE I: PERFORMANCE COMPARISON (DIRECTIONAL)")
    print("-" * 40)
    print(f"Accuracy : {acc:.4f}")
    print(f"F1-Score : {f1:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print("-" * 40)

    # --- 2. FIGURE: Confusion Matrix ---
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_val, preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Down', 'Up'], yticklabels=['Down', 'Up'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('results/confusion_matrix.png')
    plt.close()
    print("[INFO] 'confusion_matrix.png' results klasörüne kaydedildi.")

    # --- 3. FIGURE: Feature Importance ---
    if feature_names is not None:
        plt.figure(figsize=(10, 8))
        # XGBoost feature importance'ı feature_names ile eşleştir
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]

        # İlk 20 özelliği göster
        top_n = 20
        plt.title('Top 20 Feature Importances')
        plt.barh(range(top_n), importances[indices[:top_n]], align='center')
        plt.yticks(range(top_n), [feature_names[i] for i in indices[:top_n]])
        plt.xlabel('Relative Importance')
        plt.gca().invert_yaxis()  # En önemlisi en üstte olsun
        plt.savefig('results/feature_importance.png')
        plt.close()
        print("[INFO] 'feature_importance.png' results klasörüne kaydedildi.")

    # Kaydetme
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/xgb_classifier.joblib')
    return model


from tensorflow.keras.layers import Input
from tensorflow.keras.regularizers import l2


# Diğer importlar aynı...

def train_lstm_regressor(X_train_seq, y_train_seq, X_val_seq, y_val_seq, input_shape):
    """
    LSTM Regresör - Dengeli ve Optimize Edilmiş Versiyon
    Veri setine uygun boyutta, ezberlemeyi önleyen yapı.
    """
    print("\n[LSTM] Model eğitiliyor (Optimize Mod)...")

    model = Sequential()

    # Giriş Katmanı (Warning düzeltmesi)
    model.add(Input(shape=input_shape))

    # 1. Katman: Daha makul nöron sayısı + L2 Regularization
    # kernel_regularizer=l2(0.01) ağırlıkların patlamasını engeller
    model.add(LSTM(64, return_sequences=True, kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.4))  # Dropout artırıldı (%40 unutma oranı)

    # 2. Katman
    model.add(LSTM(32, return_sequences=False, kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.4))

    # Çıktı Katmanı
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1))

    # Optimizer ayarı
    from tensorflow.keras.optimizers import Adam
    opt = Adam(learning_rate=0.001)  # Standart hızda başlayalım

    model.compile(optimizer=opt, loss='mean_squared_error')

    # Callbacks
    early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001, verbose=1)

    history = model.fit(X_train_seq, y_train_seq,
                        validation_data=(X_val_seq, y_val_seq),
                        batch_size=32,  # 16 çok gürültülü olabilir, 32 standardına dönelim
                        epochs=100,
                        callbacks=[early_stop, reduce_lr],
                        verbose=1)

    # --- Değerlendirme ---
    preds = model.predict(X_val_seq)
    mse = mean_squared_error(y_val_seq, preds)
    rmse = np.sqrt(mse)

    print("-" * 40)
    print("TABLE II: REGRESSION ERROR METRICS (OPTIMIZED)")
    print("-" * 40)
    print(f"MSE      : {mse:.6f}")
    print(f"RMSE     : {rmse:.6f}")
    print("-" * 40)

    # --- Grafik ---
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('LSTM Model Loss (Optimized)')
    plt.ylabel('Mean Squared Error')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig('results/lstm_loss_curve.png')
    plt.close()

    os.makedirs('models', exist_ok=True)
    model.save('models/lstm_regressor.h5')  # .h5 yerine .keras kullanmak daha iyidir ama şimdilik kalsın
    return model