# =====================================================
# tunisia_real_estate_pipeline.py - FIXED v1.1
# =====================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from pathlib import Path
import joblib
from datetime import datetime

RAW_FILES = ['tayara_real_estate_p1_p317.csv', 'tunisia_estate_complete_clean.csv']
CLEANED_FILE = 'tunisia_estate_pipeline_clean.csv'
MODEL_FILE = 'estate_price_model.pkl'
PREDICTIONS_FILE = 'price_predictions.csv'

print("🚀 TUNISIA REAL ESTATE PIPELINE v1.1 (Fixed)")
print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

# 1. LOAD & COMBINE (FIXED SOURCE ASSIGNMENT)
print("\n📥 1. Loading...")
dfs = []
for file, source_name in zip(RAW_FILES, ['tayara', 'mubawab']):
    if Path(file).exists():
        df_temp = pd.read_csv(file)
        df_temp['source'] = source_name  # SCALAR assignment - works for all rows
        dfs.append(df_temp)
        print(f"  ✓ {file} ({len(df_temp)} rows)")
    else:
        print(f"⚠️  {file} missing")

df_raw = pd.concat(dfs, ignore_index=True)
print(f"Raw combined: {df_raw.shape}")

# 2. CLEANING & FEATURES
print("\n🧹 2. Cleaning...")
def clean_pipeline(df):
    # Price
    df = df.copy()
    df['price_numeric'] = df['price'].astype(str).str.extract('(\d[\d.,]*|\d+)')[0].str.replace(',', '').astype(float)
    df['price_numeric'] = df['price_numeric'].where(df['price_numeric'] > 0, np.nan)
    
    df['location_clean'] = df['location'].astype(str).str.title().str.strip()
    df['is_rental'] = (df['price_numeric'] < 10000).astype(int)
    
    return df.dropna(subset=['price_numeric'])

df_clean = clean_pipeline(df_raw)
print(f"Clean: {df_clean.shape}")

# 3. TRAIN MODEL
print("\n🤖 3. Modeling...")
features = ['source', 'location_clean', 'is_rental', 'page']
df_ml = pd.get_dummies(df_clean[features], drop_first=True)
X = df_ml
y = np.log1p(df_clean['price_numeric'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mae = mean_absolute_error(np.expm1(y_test), np.expm1(y_pred))
print(f"✅ MAE: {mae:,.0f} TND | Features: {X.shape[1]}")

# 4. EXPORT
df_clean.to_csv(CLEANED_FILE, index=False)
joblib.dump(model, MODEL_FILE)

print(f"\n💾 Pipeline complete!")
print(f"  Data: {CLEANED_FILE}")
print(f"  Model: {MODEL_FILE}")
print("  Predict: joblib.load('{MODEL_FILE}').predict(new_X)")

# Quick prediction example
sample_X = X.iloc[[0]].fillna(0)
pred_price = np.expm1(model.predict(sample_X)[0])
print(f"Sample prediction: {pred_price:,.0f} TND")
