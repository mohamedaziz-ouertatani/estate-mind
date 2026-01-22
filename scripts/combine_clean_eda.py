# tunisia_real_estate_analysis.py
# Complete script for combining, cleaning, and analyzing Tayara + Mubawab datasets
# Save as .py and run: python tunisia_real_estate_analysis.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# -------------------------
# CONFIG & FILEPATHS
# -------------------------
TAYARA_FILE = 'data\\raw\\tayara_real_estate_p1_p317.csv'
MUBAWAB_FILE = 'data\\raw\\tunisia_estate_complete_clean.csv'
OUTPUT_FILE = 'data\\raw\\tunisia_real_estate_combined_clean.csv'

# -------------------------
# DATA LOAD
# -------------------------
print("Loading datasets...")
df_tayara = pd.read_csv(TAYARA_FILE)
df_mubawab = pd.read_csv(MUBAWAB_FILE)

df_tayara['source'] = 'tayara'
df_mubawab['source'] = 'mubawab'

# -------------------------
# COMBINE DATA
# -------------------------
df = pd.concat([df_tayara, df_mubawab], ignore_index=True)
print(f"Combined shape: {df.shape}")

# -------------------------
# CLEANING HELPERS
# -------------------------
def clean_price(price_str):
    if pd.isna(price_str):
        return np.nan
    price_str = str(price_str).upper().replace('DT', '').replace(',', '').strip()
    price_str = ''.join(c for c in price_str if c.isdigit() or c == '.')
    try:
        return float(price_str)
    except:
        return np.nan

def clean_location(loc):
    if pd.isna(loc):
        return 'Unknown'
    return str(loc).title().strip()

# -------------------------
# CLEANING
# -------------------------
if 'price' in df.columns:
    df['price_clean'] = df['price'].apply(clean_price)
else:
    print("Column 'price' not found! Skipping price_clean.")
if 'location' in df.columns:
    df['location_clean'] = df['location'].apply(clean_location)
else:
    print("Column 'location' not found! Skipping location_clean.")
if 'category' in df.columns:
    df['category'] = df['category'].fillna('unknown')
else:
    df['category'] = 'unknown'

# -------------------------
# REMOVE DUPLICATES
# -------------------------
dedup_cols = [col for col in ['url', 'title'] if col in df.columns]
if dedup_cols:
    df_clean = df.drop_duplicates(subset=dedup_cols).reset_index(drop=True)
    print(f"After dedup: {df_clean.shape}")
else:
    df_clean = df.copy()
    print("Could not deduplicate on url/title (column not present).")

print(f"Sources: {df_clean['source'].value_counts().to_dict()}")

# -------------------------
# BASIC STATS
# -------------------------
print("\nTop locations:")
if 'location_clean' in df_clean.columns and 'price_clean' in df_clean.columns:
    top_locs = df_clean.groupby('location_clean')['price_clean'].agg(['count', 'mean']) \
        .sort_values('count', ascending=False).head(10)
    print(top_locs.round(2))
else:
    print("location_clean or price_clean not available for groupby!")

# -------------------------
# SAVE CLEANED DATA
# -------------------------
df_clean.to_csv(OUTPUT_FILE, index=False)
print(f"\nCleaned data saved: {OUTPUT_FILE}")

# -------------------------
# QUICK VISUALIZATION
# -------------------------
if 'source' in df_clean.columns and 'price_clean' in df_clean.columns:
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df_clean, x='source', y='price_clean')
    plt.title('Price Distribution by Platform')
    plt.yscale('log')
    plt.savefig('price_by_platform.png')
    plt.show()
    print("Visualization saved: price_by_platform.png")
else:
    print("Not enough columns for price_by_platform visualization.")

print("Analysis complete! Check tunisia_real_estate_combined_clean.csv and price_by_platform.png")