#!/usr/bin/env python3
"""
Mubawab Data Preparation Script v1.2 (Relaxed Filtering)
Optimized for real scraped data: Handles 0% locations, saves RAW + CLEAN CSVs.
From 8985 raw → ~4-5k valid listings with price >0.
"""

import pandas as pd
import numpy as np
import re
import glob
import os
import argparse
from pathlib import Path

def clean_price(price):
    """Extract numeric (335,000 TND → 335000.0)."""
    if pd.isna(price) or str(price).strip() == '' or str(price) == 'N/A':
        return np.nan
    match = re.search(r'(\d+(?:,\d{3})*(?:\.\d+)?)', str(price))
    return float(match.group(1).replace(',', '')) if match else np.nan

def clean_size(size):
    """Extract m² numeric."""
    if pd.isna(size) or str(size).strip() == '' or str(size) == 'N/A':
        return np.nan
    match = re.search(r'(\d+(?:,\d+)?)', str(size))
    return float(match.group(1).replace(',', '')) if match else np.nan

def clean_rooms(rooms):
    """Extract rooms numeric."""
    if pd.isna(rooms) or str(rooms).strip() == '' or str(rooms) == 'N/A':
        return np.nan
    return pd.to_numeric(str(rooms), errors='coerce')

def prepare_data(output_dir='output'):
    """Process Mubawab CSVs → raw + clean datasets."""
    Path('processed').mkdir(exist_ok=True)
    
    # Load all CSVs
    mubawab_files = glob.glob(f"{output_dir}/mubawab_max_*.csv")
    if not mubawab_files:
        raise FileNotFoundError(f"No mubawab_max_*.csv found in {output_dir}/")
    
    dfs = []
    for file in mubawab_files:
        print(f"Loading {os.path.basename(file)}")
        df = pd.read_csv(file)
        df['source'] = 'mubawab'
        dfs.append(df)
    
    full_df = pd.concat(dfs, ignore_index=True)
    print(f"✅ Merged {len(full_df):,} raw rows from {len(mubawab_files)} files.")
    
    # Dedup (safe if no URLs)
    if 'url' in full_df.columns:
        full_df['url_clean'] = full_df['url'].fillna('unknown')
        full_df.drop_duplicates(subset=['url_clean'], inplace=True)
        full_df.drop('url_clean', axis=1, inplace=True)
    
    # Clean text fields (NaN → 'N/A')
    text_cols = ['title', 'price', 'location', 'size_m2', 'rooms', 'url']
    for col in text_cols:
        if col in full_df.columns:
            full_df[col] = full_df[col].astype(str).replace('nan', 'N/A').str.strip()
    
    # Numeric features
    full_df['price_numeric'] = full_df['price'].apply(clean_price)
    full_df['size_numeric'] = full_df['size_m2'].apply(clean_size)
    full_df['rooms_numeric'] = full_df['rooms'].apply(clean_rooms)
    
    # Derived
    full_df['price_per_m2'] = np.where(
        full_df['size_numeric'] > 0,
        full_df['price_numeric'] / full_df['size_numeric'],
        np.nan
    )
    full_df['valid_price'] = full_df['price_numeric'] > 0
    
    # Categories
    category_map = {
        'apt_rent': 'Apartment Rent', 'apt_sale': 'Apartment Sale',
        'house_rent': 'House Rent', 'house_sale': 'House Sale'
    }
    if 'category' in full_df.columns:
        full_df['category_std'] = full_df['category'].map(category_map).fillna(full_df['category'])
    
    # RELAXED FILTER: Only require valid price >0 (handles 0% locations)
    valid_mask = full_df['price_numeric'].notna() & (full_df['price_numeric'] > 0)
    clean_df = full_df[valid_mask].copy()
    
    # Save RAW (all data) + CLEAN
    raw_file = 'processed/tunisia_real_estate_mubawab_raw.csv'
    full_df.to_csv(raw_file, index=False)
    
    clean_file = 'processed/tunisia_real_estate_mubawab_clean.csv'
    clean_df.to_csv(clean_file, index=False)
    
    # Stats
    print(f"\n🎉 RAW dataset: {len(full_df):,} rows → {raw_file}")
    print(f"✅ CLEAN dataset: {len(clean_df):,} listings → {clean_file}")
    print(f"💰 Avg price:     {clean_df['price_numeric'].mean():,.0f} TND")
    print(f"📏 Avg size:      {clean_df['size_numeric'].mean():.1f} m²")
    print(f"🛏️ Avg rooms:     {clean_df['rooms_numeric'].mean():.1f}")
    print(f"📊 By category:\n{clean_df['category_std'].value_counts()}")
    
    return clean_df

def main():
    parser = argparse.ArgumentParser(description="Mubawab v1.2 Prep (Relaxed)")
    parser.add_argument('--output-dir', default='output', help='Dir with CSVs')
    args = parser.parse_args()
    
    prepare_data(args.output_dir)

if __name__ == "__main__":
    main()
