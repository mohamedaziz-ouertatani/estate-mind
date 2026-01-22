#!/usr/bin/env python3
"""
COMPLETE Pipeline: Merge 4 CSVs → Clean → EDA → Model-Ready Dataset v2.0
Handles your apt_rent/sale + house_rent/sale (8985 raw → 5k+ clean).
Generates stats, plots, enhanced CSV for TDSP Phase 2.
pip install pandas numpy matplotlib seaborn scikit-learn
"""

import pandas as pd
import numpy as np
import re
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

plt.style.use('seaborn-v0_8')
Path('final').mkdir(exist_ok=True)

def robust_clean_price(price):
    """Extracts ALL prices: 600 TND, 335,000 TND → numeric."""
    if pd.isna(price) or str(price) == 'N/A': return np.nan
    match = re.search(r'(\d{1,3}(?:,\d{3})*(?:\.\d+)?)', str(price).upper())
    return float(match.group(1).replace(',', '')) if match else np.nan

def merge_and_clean():
    """Merge 4 CSVs → Enhanced dataset."""
    csv_files = glob.glob("output/mubawab_max_*.csv")
    if len(csv_files) != 4:
        print(f"Found {len(csv_files)} files; expected 4.")
    
    dfs = []
    for file in csv_files:
        df = pd.read_csv(file)
        df['source_file'] = Path(file).name
        dfs.append(df)
    
    df = pd.concat(dfs, ignore_index=True)
    print(f"📥 Merged {len(df):,} raw rows from 4 files")
    
    # Clean text
    for col in ['title', 'price', 'location', 'size_m2', 'rooms', 'url']:
        if col in df.columns:
            df[col] = df[col].astype(str).replace('nan', 'N/A').str.strip()
    
    # Numeric features
    df['price_numeric'] = df['price'].apply(robust_clean_price)
    df['size_numeric'] = pd.to_numeric(df['size_m2'].str.extract(r'(\d+)')[0], errors='coerce')
    df['rooms_numeric'] = pd.to_numeric(df['rooms'], errors='coerce')
    
    # Derived
    df['price_per_m2'] = np.where(df['size_numeric'] > 0, df['price_numeric'] / df['size_numeric'], np.nan)
    df['log_price'] = np.log1p(df['price_numeric'])
    df['is_sale'] = df['category'].str.contains('sale')
    
    # Filter valid
    valid_df = df[df['price_numeric'] > 0].copy()
    print(f"✅ {len(valid_df):,} valid listings")
    
    valid_df.to_csv('final/tunisia_estate_complete_clean.csv', index=False)
    return valid_df

def eda_analysis(df):
    """Comprehensive EDA with plots."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Listings by category
    df['category'].value_counts().plot(kind='bar', ax=axes[0,0])
    axes[0,0].set_title('Listings by Category')
    axes[0,0].tick_params('x', rotation=45)
    
    # 2. Price boxplot
    sns.boxplot(data=df, x='category', y='price_numeric', ax=axes[0,1])
    axes[0,1].set_title('Price by Category (TND)')
    axes[0,1].tick_params('x', rotation=45)
    
    # 3. Price histogram
    df['price_numeric'].hist(bins=50, ax=axes[0,2], log=True)
    axes[0,2].set_title('Price Distribution (Log)')
    
    # 4. Size vs Price scatter
    axes[1,0].scatter('size_numeric', 'price_numeric', data=df, alpha=0.5)
    axes[1,0].set_xlabel('Size m²'); axes[1,0].set_ylabel('Price TND')
    axes[1,0].set_title('Size vs Price')
    
    # 5. Rooms distribution
    df['rooms_numeric'].hist(bins=20, ax=axes[1,1])
    axes[1,1].set_title('Rooms Distribution')
    
    # 6. Price per m2
    df['price_per_m2'].hist(bins=30, ax=axes[1,2])
    axes[1,2].set_title('Price per m²')
    
    plt.tight_layout()
    plt.savefig('final/complete_eda.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Table summary
    summary = df.groupby('category')[['price_numeric', 'size_numeric', 'rooms_numeric']].agg(['count', 'mean', 'median']).round(0)
    print("\n📊 SUMMARY STATS:\n", summary)
    summary.to_csv('final/summary_stats.csv')

def quick_model(df):
    """Baseline price prediction model."""
    features = ['size_numeric', 'rooms_numeric', 'is_sale']
    df_ml = df[features + ['price_numeric']].dropna()
    
    if len(df_ml) < 100:
        print("⚠️ Insufficient data for modeling")
        return
    
    X = pd.get_dummies(df_ml[features], drop_first=True)
    y = df_ml['price_numeric']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    train_mae = mean_absolute_error(y_train, model.predict(X_train))
    test_mae = mean_absolute_error(y_test, model.predict(X_test))
    
    print(f"\n🤖 BASELINE MODEL (Random Forest)")
    print(f"Train MAE: {train_mae:,.0f} TND")
    print(f"Test MAE:  {test_mae:,.0f} TND")
    print(f"Feature Importances: {dict(zip(X.columns, model.feature_importances_.round(3)))}")

def main():
    df = merge_and_clean()
    eda_analysis(df)
    quick_model(df)
    print("\n🎉 COMPLETE! Files in 'final/': clean CSV, EDA plots, stats.")

if __name__ == "__main__":
    main()
