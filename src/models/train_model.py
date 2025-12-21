# src/models/train_model.py

from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
import joblib

# Sonuç klasörü
os.makedirs('results', exist_ok=True)
os.makedirs('models', exist_ok=True)


def create_lstm_sequences(data, target, seq_length):
    # LSTM fonksiyonu (Gerekirse kalsın, şu an kullanmıyoruz)
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length)])
        y.append(target[i + seq_length])
    return np.array(X), np.array(y)


# def train_rf_optimized(X_numeric, embeddings, y_train, X_numeric_val, embeddings_val, y_val, feature_names):
#     """
#     SelectKBest OLMADAN, Tüm özellikleri kullanan Random Forest.
#     """
#     print("\n[Random Forest] Kapsamlı eğitim başlıyor (Tüm Özellikler)...")
#
#     # --- ADIM 1: Embedding Sıkıştırma (PCA) ---
#     # 768 boyutu yine de biraz indirelim ki model boğulmasın.
#     n_components = 15
#     pca = PCA(n_components=n_components)
#
#     print(f"BERT Embedding'leri PCA ile {n_components} boyuta indiriliyor...")
#     X_emb_train_pca = pca.fit_transform(embeddings)
#     X_emb_val_pca = pca.transform(embeddings_val)
#
#     pca_cols = [f"PCA_Emb_{i}" for i in range(n_components)]
#
#     # --- ADIM 2: Verileri Birleştirme ---
#     # Sayısal veriler
#     if isinstance(X_numeric, pd.DataFrame):
#         X_num_train = X_numeric.values
#         X_num_val = X_numeric_val.values
#     else:
#         X_num_train = X_numeric
#         X_num_val = X_numeric_val
#
#     # Hepsini yan yana koy
#     X_train_combined = np.hstack([X_num_train, X_emb_train_pca])
#     X_val_combined = np.hstack([X_num_val, X_emb_val_pca])
#
#     # Tüm isimler
#     all_feature_names = feature_names + pca_cols
#
#     print(f"Toplam Özellik Sayısı: {len(all_feature_names)}")
#
#     # --- ADIM 3: Model Eğitimi (Feature Selection YOK) ---
#     # Kararı Random Forest'a bırakıyoruz.
#     rf = RandomForestClassifier(
#         n_estimators=500,  # Ağaç sayısını artırdık
#         max_depth=7,  # Derinliği biraz artırdık (Daha karmaşık ilişkiler için)
#         min_samples_leaf=4,
#         random_state=42,
#         n_jobs=-1,
#         class_weight='balanced'  # Korkaklığı önlemek için
#     )
#
#     rf.fit(X_train_combined, y_train)
#
#     # --- Değerlendirme ---
#     preds = rf.predict(X_val_combined)
#     acc = accuracy_score(y_val, preds)
#
#     print("-" * 40)
#     print(f"RANDOM FOREST ACCURACY: {acc:.4f}")
#     print("-" * 40)
#     print(classification_report(y_val, preds))
#
#     # --- Feature Importance Grafiği ---
#     importances = rf.feature_importances_
#     indices = np.argsort(importances)[::-1]
#
#     # İlk 20'yi çiz
#     top_n = 20
#     plt.figure(figsize=(10, 8))
#     plt.title('Top 20 Most Important Features')
#     plt.barh(range(top_n), importances[indices[:top_n]], align='center')
#     plt.yticks(range(top_n), [all_feature_names[i] for i in indices[:top_n]])
#     plt.xlabel('Relative Importance')
#     plt.gca().invert_yaxis()
#     plt.tight_layout()
#     plt.savefig('results/rf_feature_importance_full.png')
#     plt.close()
#
#     joblib.dump(rf, 'models/rf_optimized.joblib')
#     return rf


# # Dosya: src/models/train_model.py
# import pandas as pd
# import numpy as np
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import classification_report, accuracy_score


def train_rf_optimized(X_train, y_train, X_val, y_val, feature_names=None):
    """
    Random Forest modelini eğitir ve geri döndürür.
    X_train artık hem sayısal verileri hem de embeddingleri içeren birleşik matristir.
    """
    print(f"Model Eğitiliyor... (Girdi Boyutu: {X_train.shape})")

    # Basit ve etkili hiperparametreler
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1,  # Tüm işlemcileri kullan
        class_weight='balanced'
    )

    clf.fit(X_train, y_train)

    print("\n--- Validation Sonuçları ---")
    val_preds = clf.predict(X_val)
    print(classification_report(y_val, val_preds))
    print(f"Accuracy: {accuracy_score(y_val, val_preds):.4f}")

    return clf