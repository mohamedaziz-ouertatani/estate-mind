#!/usr/bin/env python3
"""
Mubawab RAW EDA & Validation v1.0
Unlocks all 8985 rows: fixes price parsing, stats by category, visualizations prep.
Save as eda_mubawab.py; run `python eda_mubawab.py`
"""

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import glob

plt.style.use('seaborn-v0_8')
Path('eda').mkdir(exist_ok=True)

def robust_price_parse(price):
    """Aggressive regex for ALL price formats from diagnostics."""
    if pd.isna(price) or str(price) == 'N/A':
        return np.nan
    price_str = str(price).upper()
    # Handles 600 TND, 335,000 TND, 2,800 TND, etc.
    match = re.search(r'(\d{1,3}(?:,\d{3})*(?:\.\d+)?)', price_str)
    return float(match.group(1).replace(',', '')) if match else np.nan

# Load RAW data (8985 rows)
raw_files = glob.glob("data\\raw\\tunisia_real_estate_mubawab_raw.csv")
if not raw_files:
    raw_files = glob.glob("data\\raw\\mubawab_max_*.csv")
if not raw_files:
    print("No raw CSV found. Run 00_prepare_mubawab.py first!")
    exit()

df_list = []
for f in raw_files:
    temp_df = pd.read_csv(f)
    temp_df['source_file'] = Path(f).name
    df_list.append(temp_df)

eda_df = pd.concat(df_list, ignore_index=True)
print(f"Loaded {len(eda_df)} raw rows")

# Enhanced cleaning
eda_df['price_robust'] = eda_df['price'].apply(robust_price_parse)
eda_df['valid_price'] = eda_df['price_robust'] > 0

# Stats
valid_df = eda_df[eda_df['valid_price']].copy()
print(f"\n✅ VALID LISTINGS: {len(valid_df):,} (was 1, now unlocked!)")
print(f"💰 Avg Price: {valid_df['price_robust'].mean():,.0f} TND")
print(f"📊 By Category:\n{valid_df.groupby('category')['price_robust'].agg(['count', 'mean', 'median']).round(0)}")

# Save enhanced clean
valid_df.to_csv('eda/tunisia_real_estate_enhanced.csv', index=False)
print("💾 Saved enhanced CSV")

# Quick Plots
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Price dist by category
sns.boxplot(data=valid_df, x='category', y='price_robust', ax=axes[0,0])
axes[0,0].set_title('Price Distribution by Category')
axes[0,0].tick_params(axis='x', rotation=45)

# 2. Count by category
valid_df['category'].value_counts().plot(kind='bar', ax=axes[0,1])
axes[0,1].set_title('Listings per Category')

# 3. Price histogram
valid_df['price_robust'].hist(bins=50, ax=axes[1,0], log=True)
axes[1,0].set_title('Price Histogram (Log Scale)')

# 4. Price per m2 (where available)
if 'size_numeric' in valid_df.columns:
    valid_df['price_per_m2'] = valid_df['price_robust'] / valid_df['size_numeric'].replace(0, np.nan)
    valid_df['price_per_m2'].hist(bins=30, ax=axes[1,1])
    axes[1,1].set_title('Price per m²')

plt.tight_layout()
plt.savefig('eda/mubawab_eda.png', dpi=300, bbox_inches='tight')
plt.show()

print("📈 EDA plots saved to eda/mubawab_eda.png")
print("🚀 Dataset ready for modeling! Next: estate_pipeline.py")
