import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Veri setlerini yükleme (dosya adlarını gerçek dosya adlarınızla değiştirin)
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Eğitim verilerini inceleme
print("Eğitim veri seti şekli:", train_data.shape)
print(train_data.head())

# Hedef değişkeni (price_range) inceleme
print("\nHedef değişken (price_range) değer dağılımı:")
print(train_data['price_range'].value_counts())

# Özelliklerin korelasyonunu inceleme
plt.figure(figsize=(16, 12))
correlation_matrix = train_data.corr()
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm')
plt.title('Özellikler Arası Korelasyon Matrisi')
plt.tight_layout()
plt.savefig('correlation_matrix.png')

# Hedef değişkenle en çok ilişkili özellikleri bulma
target_correlations = correlation_matrix['price_range'].sort_values(ascending=False)
print("\nHedef değişkenle en yüksek korelasyona sahip özellikler:")
print(target_correlations)

# Eğitim verilerini hazırlama
X_train = train_data.drop('price_range', axis=1)
y_train = train_data['price_range']

# Test verilerini hazırlama (test veri setindeki 'id' sütununu çıkaralım)
X_test = test_data.drop('id', axis=1) if 'id' in test_data.columns else test_data

# Veri normalizasyonu
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Eğitim verilerini eğitim ve doğrulama setlerine ayırma
X_train_split, X_val, y_train_split, y_val = train_test_split(
    X_train_scaled, y_train, test_size=0.2, random_state=42
)

# 1. Rastgele Orman Sınıflandırıcısı
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train_split, y_train_split)
rf_val_pred = rf_model.predict(X_val)
rf_val_accuracy = accuracy_score(y_val, rf_val_pred)
print(f"\nRastgele Orman Doğrulama Doğruluğu: {rf_val_accuracy:.4f}")

# 2. Gradient Boosting Sınıflandırıcısı
gb_model = GradientBoostingClassifier(random_state=42)
gb_model.fit(X_train_split, y_train_split)
gb_val_pred = gb_model.predict(X_val)
gb_val_accuracy = accuracy_score(y_val, gb_val_pred)
print(f"Gradient Boosting Doğrulama Doğruluğu: {gb_val_accuracy:.4f}")

# 3. Lojistik Regresyon
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_split, y_train_split)
lr_val_pred = lr_model.predict(X_val)
lr_val_accuracy = accuracy_score(y_val, lr_val_pred)
print(f"Lojistik Regresyon Doğrulama Doğruluğu: {lr_val_accuracy:.4f}")


#****************************************************


# En iyi performansa sahip modeli seçme ve hiperparametre optimizasyonu yapma
# Rastgele Orman için hiperparametre arama
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring='accuracy'
)
grid_search.fit(X_train_scaled, y_train)

print("\nEn iyi Rastgele Orman parametreleri:")
print(grid_search.best_params_)
print(f"En iyi çapraz doğrulama skoru: {grid_search.best_score_:.4f}")

# En iyi modeli tam eğitim veri setiyle eğitme
best_model = grid_search.best_estimator_
best_model.fit(X_train_scaled, y_train)

# Özellik önemini inceleme
feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': best_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nÖzellik önemleri:")
print(feature_importance.head(10))

plt.figure(figsize=(12, 8))
sns.barplot(x='importance', y='feature', data=feature_importance.head(10))
plt.title('En Önemli 10 Özellik')
plt.tight_layout()
plt.savefig('feature_importance.png')

# Test veri seti üzerinde tahminler yapma
test_predictions = best_model.predict(X_test_scaled)

# Eğer test veri setinde gerçek etiketler varsa, modelin performansını değerlendirebilirsiniz
# Bu durumda sadece tahminleri kaydediyoruz

# Test tahminlerini kaydetme
result_df = pd.DataFrame({
    'id': test_data['id'] if 'id' in test_data.columns else range(len(X_test)),
    'predicted_price_range': test_predictions
})
result_df.to_csv('predictions.csv', index=False)

print("\nTahminler 'predictions.csv' olarak kaydedildi.")

# Confusion matrix ve sınıflandırma raporu oluşturma
# Doğrulama seti üzerinden
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_val, rf_val_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix (Doğrulama Seti)')
plt.ylabel('Gerçek Değer')
plt.xlabel('Tahmin Edilen Değer')
plt.savefig('confusion_matrix.png')

print("\nSınıflandırma Raporu (Doğrulama Seti):")
print(classification_report(y_val, rf_val_pred))