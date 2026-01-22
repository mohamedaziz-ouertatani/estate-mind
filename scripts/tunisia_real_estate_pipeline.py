# =============================================================
# tunisia_real_estate_pipeline_advanced.py
# All-in-one: Outlier drop, feature importance, feature-rich,
# plus: Rent/sale classifier! (robust for empty classifier data)
# =============================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, classification_report
from pathlib import Path
import joblib
from datetime import datetime

RAW_FILES = [
    'data\\raw\\tayara_real_estate_p1_p317.csv',
    'data\\raw\\tunisia_estate_complete_clean.csv'
]
CLEANED_FILE = 'tunisia_estate_pipeline_clean.csv'
MODEL_FILE = 'estate_price_model.pkl'
CLASSIFIER_FILE = 'estate_rentsale_classifier.pkl'
PREDICTIONS_FILE = 'price_predictions.csv'

print("🚀 TUNISIA REAL ESTATE PIPELINE ADVANCED v1.2")
print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

# 1. LOAD & COMBINE
print("\n📥 1. Loading...")
dfs = []
for file, source_name in zip(RAW_FILES, ['tayara', 'mubawab']):
    if Path(file).exists():
        df_temp = pd.read_csv(file)
        df_temp['source'] = source_name
        dfs.append(df_temp)
        print(f"  ✓ {file} ({len(df_temp)} rows)")
    else:
        print(f"⚠️  {file} missing")
if not dfs:
    raise FileNotFoundError("No input data files found. Check your RAW_FILES paths.")

df_raw = pd.concat(dfs, ignore_index=True)
print(f"Raw combined: {df_raw.shape}")

# 2. CLEANING, MORE FEATURES, OUTLIER REMOVAL
print("\n🧹 2. Cleaning & Feature Engineering...")

def parse_float(s):
    try:
        s = str(s).replace(',', '').replace('DT', '').strip()
        return float(''.join(c for c in s if (c.isdigit() or c == '.')))
    except:
        return np.nan

def clean_pipeline(df):
    df = df.copy()
    df['price_numeric'] = (
        df['price']
        .astype(str)
        .str.extract(r'(\d[\d.,]*|\d+)')[0]
        .str.replace(',', '')
        .astype(float)
    )
    df['price_numeric'] = df['price_numeric'].where(df['price_numeric'] > 0, np.nan)
    # Add ANY extra numerics available (if present)
    for feat in ['size', 'size_m2', 'surface', 'area', 'rooms', 'rooms_numeric', 'bedrooms']:
        if feat in df.columns:
            col = f"{feat}_num" if not feat.endswith('_num') else feat
            df[col] = df[feat].apply(parse_float)
    # Standardized numerics
    numeric_features = [col for col in df.columns if any(x in col for x in ['size', 'surface', 'area', 'room', 'bedroom'])]
    df['location_clean'] = df['location'].astype(str).str.title().str.strip()
    # Robust rent/sale detection
    if 'category' in df.columns:
        df['deal_type'] = df['category'].astype(str).str.lower().apply(
            lambda x: 'rent' if 'rent' in x or 'louer' in x or 'location' in x
            else ('sale' if 'sale' in x or 'vente' in x else 'unknown')
        )
    elif 'type' in df.columns:
        df['deal_type'] = df['type'].astype(str).str.lower()
    else:
        df['deal_type'] = np.where(df['price_numeric'] < 10000, 'rent', 'sale')
    # Remove crazy price outliers (99th percentile)
    max_price = df['price_numeric'].quantile(0.99)
    min_price = 100  # 100 TND sanity check
    df2 = df[(df['price_numeric'] < max_price) & (df['price_numeric'] > min_price)].copy()
    return df2.dropna(subset=['price_numeric', 'deal_type'])

df_clean = clean_pipeline(df_raw)
print(f"Clean: {df_clean.shape}")
print(f"Deals: {df_clean['deal_type'].value_counts().to_dict()}")

# 3. RENT/SALE CLASSIFIER
print("\n🔎 3. Rent vs. Sale Classifier")
label_map = {'rent': 1, 'sale': 0}
df_clean = df_clean[df_clean['deal_type'].isin(['rent', 'sale'])]
y_class = df_clean['deal_type'].map(label_map)

cls_features = ['source', 'location_clean']
extra_cls_features = [
    col for col in df_clean.columns
    if any(x in col for x in ['size', 'surface', 'area', 'room', 'bedroom'])
    and col not in cls_features
]
cls_features = cls_features + extra_cls_features
cls_X = pd.get_dummies(df_clean[cls_features])

print(f"Train classifier: cls_X shape: {cls_X.shape}, y_class shape: {y_class.shape}")
if cls_X.empty or y_class.empty:
    print("⚠️ Classifier training data is empty after cleaning. Skipping classifier training.")
    cls = None
else:
    if len(y_class.unique()) < 2:
        print("⚠️ Only one class present in deal_type after cleaning. Skipping classifier training.")
        cls = None
    else:
        X_cls_train, X_cls_test, y_cls_train, y_cls_test = train_test_split(
            cls_X, y_class, test_size=0.2, random_state=42
        )
        if X_cls_train.empty or y_cls_train.empty:
            print("⚠️ Classifier train set is empty after train_test_split; skipping classifier training.")
            cls = None
        else:
            cls = RandomForestClassifier(n_estimators=100, random_state=42)
            cls.fit(X_cls_train, y_cls_train)
            y_cls_pred = cls.predict(X_cls_test)
            cls_acc = accuracy_score(y_cls_test, y_cls_pred)
            print(f"Rent/Sale classifier accuracy: {cls_acc:.2%}")
            print(classification_report(y_cls_test, y_cls_pred, target_names=['Sale', 'Rent']))
            joblib.dump(cls, CLASSIFIER_FILE)

# 4. REGRESSION MODEL
print("\n🤖 4. Price Regression Model (RandomForestRegressor)")
reg_base_features = ['source', 'location_clean']
feature_candidates = [
    col for col in df_clean.columns
    if any(s in col for s in ['size', 'surface', 'area', 'room', 'bedroom']) and col not in reg_base_features
]
all_features = reg_base_features + feature_candidates

df_reg_ml = pd.get_dummies(df_clean[all_features])
for col in feature_candidates:
    if col in df_reg_ml.columns:
        df_reg_ml[col] = df_reg_ml[col].fillna(-1)

X = df_reg_ml
y = np.log1p(df_clean['price_numeric'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

if X_train.empty or y_train.empty:
    raise ValueError("Regression train set is empty after train_test_split; cannot train!")

model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mae = mean_absolute_error(np.expm1(y_test), np.expm1(y_pred))
r2 = r2_score(np.expm1(y_test), np.expm1(y_pred))
print(f"✅ MAE: {mae:,.0f} TND | R2: {r2:.3f} | Features: {X.shape[1]}")

# 5. FEATURE IMPORTANCE
print("\n🌟 Feature Importances (Regression, top 15):")
importances = pd.Series(model.feature_importances_, index=X.columns)
print(importances.sort_values(ascending=False).head(15))

# 6. SAVE EVERYTHING
df_clean.to_csv(CLEANED_FILE, index=False)
joblib.dump(model, MODEL_FILE)

if cls is not None:
    print(f"Classifier saved: {CLASSIFIER_FILE}")
print(f"Regression model saved: {MODEL_FILE}")
print(f"Clean data saved: {CLEANED_FILE}")

# Sample regression prediction example
if not X.empty:
    sample_X = X.iloc[[0]].fillna(-1)
    pred_price = np.expm1(model.predict(sample_X)[0])
    print(f"Sample regression prediction: {pred_price:,.0f} TND")
# Sample classification prediction example
if cls is not None and not cls_X.empty:
    sample_cls_X = cls_X.iloc[[0]]
    pred_cls = cls.predict(sample_cls_X)[0]
    print(f"Sample classifier prediction (0=sale, 1=rent): {pred_cls}")