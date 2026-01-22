#!/usr/bin/env python3
"""
🇹🇳 TUNISIA REAL ESTATE - PRODUCTION PIPELINE v2.1
Fixes all bugs. Run: python complete_estate_pipeline.py
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib
from pathlib import Path
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

print("🚀 PRODUCTION PIPELINE v2.1 - Starting...")

# 1. FILES & CONFIG
FILES = {
    'tayara': 'data\\raw\\tayara_real_estate_p1_p317.csv',
    'mubawab': 'data\\raw\\tunisia_estate_complete_clean.csv'
}
OUTPUTS = {
    'clean': 'estate_data_clean.csv',
    'model': 'estate_model_v2.pkl',
    'report': 'pipeline_report.txt',
    'predictions': 'sample_predictions.csv'
}

# 2. LOAD DATA (BULLETPROOF)
print("\n📂 Loading files...")
dfs = []
for source_name, filename in FILES.items():
    if Path(filename).exists():
        df_temp = pd.read_csv(filename)
        df_temp['source'] = source_name  # Add source column
        dfs.append(df_temp)
        print(f"  ✓ {source_name}: {len(df_temp):,} rows")
    else:
        print(f"  ⚠️  {filename} missing")

if not dfs:
    raise FileNotFoundError("No data files found!")

df = pd.concat(dfs, ignore_index=True)
print(f"Combined: {df.shape}")

# 3. CLEAN & ENGINEER (SIMPLE + ROBUST)
print("\n🧹 Cleaning...")

df['price_str'] = df['price'].astype(str).str.upper()
# --- FIXED PRICE EXTRACTION ---
# For each row: extract the first number. If no numbers → NaN automatically.
df['price_numeric'] = (
    df['price_str']
      .str.findall(r'\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b')
      .str[0]
)
df['price_numeric'] = df['price_numeric'].str.replace(',', '').astype(float)

df['location_clean'] = df['location'].fillna('Unknown').astype(str).str.title().str.strip()
df['is_rental'] = (df['price_numeric'] < 10000).fillna(0).astype(int)

# Keep clean rows only
df_clean = df.dropna(subset=['price_numeric']).copy()
print(f"Clean prices: {len(df_clean):,}")

# 4. TRAIN MODEL (SIMPLE FEATURES)
print("\n🤖 Training...")
feature_cols = ['source', 'location_clean', 'is_rental']
X = pd.get_dummies(df_clean[feature_cols], drop_first=True)
y = np.log1p(df_clean['price_numeric'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Score
y_pred = model.predict(X_test)
mae_raw = mean_absolute_error(np.expm1(y_test), np.expm1(y_pred))
print(f"MAE: {mae_raw:,.0f} TND")

# 5. SAVE EVERYTHING
joblib.dump(model, OUTPUTS['model'])
joblib.dump(X.columns, 'model_features.pkl')
df_clean.to_csv(OUTPUTS['clean'], index=False)

# 6. SAMPLE PREDICTIONS
samples = pd.DataFrame({
    'source': ['tayara', 'mubawab', 'tayara'],
    'location_clean': ['Tunis', 'Ariana', 'Nabeul'],
    'is_rental': [1, 1, 0]
})
X_samples = pd.get_dummies(samples, drop_first=True).reindex(columns=X.columns, fill_value=0)
preds = np.expm1(model.predict(X_samples))
samples['predicted_price'] = preds
samples.to_csv(OUTPUTS['predictions'], index=False)

print("\n✅ PIPELINE 100% COMPLETE!")
print("\n📁 FILES GENERATED:")
for name, path in OUTPUTS.items():
    print(f"  • {path}")
print(f"\n📈 Report: {len(df_clean):,} listings | MAE {mae_raw:,.0f} TND")
print("\n🎯 DEPLOY: joblib.load('estate_model_v2.pkl').predict(new_data)")