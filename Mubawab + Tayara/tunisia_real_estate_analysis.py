# tunisia_real_estate_analysis.py
# Complete script for combining, cleaning, and analyzing Tayara + Mubawab datasets
# Save as .py and run: python tunisia_real_estate_analysis.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# File paths
TAYARA_FILE = 'tayara_real_estate_p1_p317.csv'
MUBAWA_FILE = 'tunisia_estate_complete_clean.csv'
OUTPUT_FILE = 'tunisia_real_estate_combined_clean.csv'

print("Loading datasets...")
df_tayara = pd.read_csv(TAYARA_FILE)
df_mubawab = pd.read_csv(MUBAWA_FILE)

# Add source labels
df_tayara['source'] = 'tayara'
df_mubawab['source'] = 'mubawab'

# Combine
df = pd.concat([df_tayara, df_mubawab], ignore_index=True)
print(f"Combined shape: {df.shape}")

# Cleaning functions
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

# Apply cleaning
df['price_clean'] = df['price'].apply(clean_price)
df['location_clean'] = df['location'].apply(clean_location)
df['category'] = df['category'].fillna('unknown')

# Remove duplicates
df_clean = df.drop_duplicates(subset=['url', 'title']).reset_index(drop=True)
print(f"After dedup: {df_clean.shape}")
print(f"Sources: {df_clean['source'].value_counts().to_dict()}")

# Basic stats
print("\nTop locations:")
top_locs = df_clean.groupby('location_clean')['price_clean'].agg(['count', 'mean']).sort_values('count', ascending=False).head(10)
print(top_locs.round(2))

# Save cleaned data
df_clean.to_csv(OUTPUT_FILE, index=False)
print(f"\nCleaned data saved: {OUTPUT_FILE}")

# Quick visualization
plt.figure(figsize=(10, 6))
sns.boxplot(data=df_clean, x='source', y='price_clean')
plt.title('Price Distribution by Platform')
plt.yscale('log')
plt.savefig('price_by_platform.png')
plt.show()

print("Analysis complete! Check tunisia_real_estate_combined_clean.csv and price_by_platform.png")
