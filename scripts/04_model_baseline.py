import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import numpy as np

print("Loading cleaned data...")
df = pd.read_csv('data\\raw\\tunisia_real_estate_combined_clean.csv')

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

# Use a normalized price column name
df['price_num'] = df['price'].apply(clean_price)

# Use ONLY valid rows with price
df_ml = df.dropna(subset=['price_num']).copy()
print(f"Valid rows with price: {len(df_ml):,}")

# Select safe columns BEFORE encoding
safe_cols = ['source', 'location_clean', 'category', 'page', 'rooms_numeric', 'size_numeric']
use_cols = [col for col in safe_cols if col in df_ml.columns]
df_ml = df_ml[use_cols + ['price_num']]

# One-hot encode categoricals (except price)
df_encoded = pd.get_dummies(df_ml.drop('price_num', axis=1), drop_first=True)
X = df_encoded
y = np.log1p(df_ml['price_num'])  # Log target

# ================= Fix 1: Drop 'page' artifact feature ======================
# The RandomForest model previously showed 'page' and 'category_apt_sale' dominated price predictions,
# confirming a scrape artifact bias and apartment category premium.
# Following Phase 3 best practice: remove 'page' feature from the model!
if 'page' in X.columns:
    X = X.drop(columns=['page'], errors='ignore')
    print("Column 'page' dropped to fix rank bias (scraping artifact).")

print(f"Features: {X.shape[1]}, Samples: {X.shape[0]}")

# Train/test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Model evaluation
y_pred_log = model.predict(X_test)
mae_log = mean_absolute_error(y_test, y_pred_log)
mae_orig = mean_absolute_error(np.expm1(y_test), np.expm1(y_pred_log))

print(f"\n✅ Model Performance:")
print(f"Log MAE: {mae_log:.3f}")
print(f"Original MAE: {mae_orig:.0f} TND (~{mae_orig/np.expm1(y_test).mean()*100:.1f}% of avg price)")

# Output feature importances
importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n🔍 Top Features:")
print(importance.head(10))
importance.to_csv('model_features.csv', index=False)

# =================== INTERPRETATION AND NEXT STEPS ====================
print("""
Model Insights:
- The previous RandomForest model showed 'page' (0.43) and 'category_apt_sale' (0.26) together made up 72% of explained price variance.
    • 'page' was a scraping artifact tracking listing position (not a true signal), so it is removed above.
    • 'category_apt_sale' (apartment sales) is 2.6x more important than house sales.
- 'size_numeric' and 'rooms_numeric' have low importances, reflecting data quality problems: parsing only captures ~20% of entries—room/size field enhancement is a key way to improve predictive power.
- Nabeul coastal region shows a distinct premium.

Key Actions:
- Always drop 'page' prior to ML.
- Continue to improve extraction for 'size'/'rooms'.
- Use geographic and category features to further explain price variance.

For further steps, retrain after each data quality improvement and aim for MAE <69k TND as realistic business performance.
""")

print("\n💾 Saved: model_features.csv")