import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import numpy as np

print("Loading cleaned data...")
df = pd.read_csv('tunisia_real_estate_combined_clean.csv')

# Ultra-strict price cleaning (handles all cases)
def clean_price(price):
    if pd.isna(price):
        return np.nan
    s = str(price).upper()
    s = ''.join(c for c in s if c.isdigit() or c == '.')
    try:
        p = float(s)
        return p if 0 < p < 10000000 else np.nan  # Sanity bounds
    except:
        return np.nan

df['price_numeric'] = df['price'].apply(clean_price)

# Use ONLY valid rows with price
df_ml = df.dropna(subset=['price_numeric']).copy()
print(f"Valid rows with price: {len(df_ml):,}")

# Select safe columns BEFORE encoding
safe_cols = ['source', 'location_clean', 'category', 'page', 'rooms_numeric', 'size_numeric']
use_cols = [col for col in safe_cols if col in df_ml.columns]
df_ml = df_ml[use_cols + ['price_numeric']]

# Encode categoricals
df_encoded = pd.get_dummies(df_ml.drop('price_numeric', axis=1), drop_first=True)
X = df_encoded
y = np.log1p(df_ml['price_numeric'])  # Log target

print(f"Features: {X.shape[1]}, Samples: {X.shape[0]}")

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred_log = model.predict(X_test)
mae_log = mean_absolute_error(y_test, y_pred_log)
mae_orig = mean_absolute_error(np.expm1(y_test), np.expm1(y_pred_log))

print(f"\n✅ Model Performance:")
print(f"Log MAE: {mae_log:.3f}")
print(f"Original MAE: {mae_orig:.0f} TND (~{mae_orig/np.expm1(y_test).mean()*100:.1f}% of avg price)")

# Feature importance (top 10)
importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False).head(10)

print("\n🔍 Top Features:")
print(importance)
importance.to_csv('model_features.csv', index=False)

print("\n💾 Saved: model_features.csv")
