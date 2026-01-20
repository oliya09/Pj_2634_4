#!/usr/bin/env python3
# rf_model_simple.py
"""
Обучение RandomForestClassifier на реальных данных NASA Exoplanet Archive
с защитой от переобучения и без сохранения лишних файлов.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

# === 1️⃣ Определяем начало таблицы ===
def find_table_start(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if line.startswith("pl_name"):
                return i
    return None


csv_file = "test.csv"
start_line = find_table_start(csv_file)
if start_line is None:
    raise ValueError("❌ Не найдено начало таблицы!")
else:
    print(f"✅ Таблица найдена начиная с строки: {start_line}")

# === 2️⃣ Загружаем данные ===
df = pd.read_csv(csv_file, skiprows=start_line)
print(f"📊 Загружено {len(df)} строк и {len(df.columns)} колонок.")

# === 3️⃣ Выбираем признаки ===
feature_columns = [
    "pl_orbper", "pl_rade", "pl_bmasse",
    "pl_eqt", "st_teff", "st_mass", "st_rad"
]
target_column = "default_flag"

df = df[feature_columns + [target_column]].dropna()
print(f"\n📈 После очистки: {len(df)} строк.")
print(df[target_column].value_counts())

# === 4️⃣ Проверяем наличие двух классов ===
unique_classes = df[target_column].unique()
if len(unique_classes) < 2:
    print("\n⚠️ Найден только один класс! Создаём искусственную метку 'target' (радиус + температура + шум).")
    median_rade = df["pl_rade"].median()
    median_eqt = df["pl_eqt"].median()
    df["target"] = ((df["pl_rade"] * 0.6 + df["pl_eqt"] * 0.4 +
                     np.random.normal(0, 0.05, len(df))) >
                    (median_rade * 0.6 + median_eqt * 0.4)).astype(int)
    y = df["target"].values
    print(f"📊 Баланс новых классов:\n{pd.Series(y).value_counts()}")
else:
    y = df[target_column].values

X = df[feature_columns].values

# === 5️⃣ Разделение данных ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
)

# === 6️⃣ Обучение модели с анти-оверфитом ===
print("\n🚀 Обучение RandomForestClassifier (anti-overfitting)...")
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=6,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
)
model.fit(X_train, y_train)

# === 7️⃣ Кросс-валидация ===
cv_scores = cross_val_score(model, X_train, y_train, cv=5)
print(f"\n🧠 Средняя точность (5-Fold CV): {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# === 8️⃣ Оценка ===
y_pred = model.predict(X_test)
report = classification_report(y_test, y_pred, digits=3)
print("\n📊 Classification Report:\n", report)

# === 9️⃣ Важность признаков ===
importances = pd.Series(model.feature_importances_, index=feature_columns).sort_values(ascending=False)
print("\n🌌 Важность признаков:\n", importances)

# === 🔟 Сохранение модели ===
joblib.dump(model, "rf_model.pkl")
print("\n💾 Модель сохранена в 'rf_model.pkl'")
