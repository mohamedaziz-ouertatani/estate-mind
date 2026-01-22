#!/usr/bin/env python3
"""
Estate-Mind Phase 2 Pipeline: Tayara + Mubawab → Clean/EDA/Report (FULLY FIXED)
Robust parsing + error-proof plotting

python estate_pipeline.py → Phase 2 submission ready!
"""

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')

# Setup
Path("output").mkdir(exist_ok=True, parents=True)
plt.style.use('default')
print("🏠 Estate-Mind Phase 2 Pipeline (Production)")

# =============================================================================
# 1. LOAD DATA (Robust)
# =============================================================================
def load_all_data():
    files = {
        'tayara': 'tayara_real_estate_p1_p317.csv',
        'mubawab_sales': 'mubawab_sales_full.csv',
        'mubawab_rentals': 'mubawab_rentals_full.csv'
    }
    sources = {}
    
    for name, fname in files.items():
        try:
            df = pd.read_csv(fname)
            sources[name] = df.assign(source=name)
            print(f"✅ {name}: {len(df)} rows")
        except FileNotFoundError:
            print(f"⚠️  {fname} skipped")
    
    if not sources:
        raise ValueError("No CSV files found!")
    
    df_raw = pd.concat(sources.values(), ignore_index=True)
    print(f"\n📊 RAW: {len(df_raw)} rows | Columns: {list(df_raw.columns)}")
    return df_raw

df = load_all_data()

# =============================================================================
# 2. CLEANING (Bulletproof)
# =============================================================================
def clean_estate_data(df):
    df_clean = df.copy()
    
    # Price parsing (handles ALL formats)
    def parse_price(price_str):
        if pd.isna(price_str): return np.nan
        s = str(price_str).upper().replace('\n', ' ').replace('\t', ' ').strip()
        nums = re.findall(r'[\d,\.]+', s.replace(',', ''))
        if nums:
            # Join digits (335,000 → 335000)
            num_str = ''.join(nums)[:12]
            try:
                return float(num_str)
            except:
                return np.nan
        return np.nan
    
    df_clean['price_num'] = df_clean['price'].apply(parse_price)
    print(f"💰 Prices: {df_clean['price_num'].notna().sum()}/{len(df_clean)}")
    
    # Type detection
    df_clean['type'] = 'unknown'
    sale_pat = r'\b(sale|vendre|VENDE|vente|buy|ACHAT)\b'
    rent_pat = r'\b(rent|louer|LOUER|location|monthly|mois)\b'
    
    df_clean.loc[df_clean['title'].str.contains(sale_pat, regex=True, case=False, na=False), 'type'] = 'sale'
    df_clean.loc[df_clean['title'].str.contains(rent_pat, regex=True, case=False, na=False), 'type'] = 'rent'
    
    print("Types:", df_clean['type'].value_counts().to_dict())
    
    # Bedrooms
    bed_pat = r'(\d+)\s*(?:room|rooms|bed|chambre|pièces?|S\d|bedroom)\b'
    df_clean['bedrooms'] = df_clean['title'].str.extract(bed_pat, flags=re.I, expand=False).astype('Int64')
    
    # Size (check all possible columns)
    size_candidates = ['size_m2', 'size', 'surface', 'area']
    size_cols = [col for col in size_candidates if col in df_clean.columns]
    if size_cols:
        size_text = df_clean[size_cols].bfill(axis=1).iloc[:, 0].fillna('').astype(str)
    else:
        size_text = ''
    df_clean['size_m2'] = size_text.str.extract(r'(\d+)').astype(float)
    
    print(f"📏 Size: {df_clean['size_m2'].notna().sum()} found")
    print(f"🛏️  Bedrooms: {df_clean['bedrooms'].notna().sum()} found")
    
    # Location
    loc_col = next((col for col in ['location', 'city', 'ville'] if col in df_clean.columns), None)
    if loc_col:
        df_clean['location_clean'] = df_clean[loc_col].astype(str).str.strip().str.title()
    
    # VALID filter
    valid_mask = (df_clean['price_num'] > 0) & (df_clean['type'] != 'unknown')
    df_valid = df_clean[valid_mask].copy()
    
    print(f"\n✅ FINAL VALID: {len(df_valid)}/{len(df)}")
    print(f"  → Sales: {len(df_valid[df_valid.type=='sale'])}")
    print(f"  → Rent:  {len(df_valid[df_valid.type=='rent'])}")
    
    return df_valid

df_clean = clean_estate_data(df)
df_clean.to_csv('output/estate_data_cleaned.csv', index=False)

# =============================================================================
# 3. EDA STATS
# =============================================================================
def eda_stats(df_clean):
    print("\n📈 STATISTICS")
    
    if len(df_clean) == 0:
        print("No valid data for EDA")
        return pd.DataFrame(), pd.DataFrame(), pd.Series()
    
    # Type/Source matrix
    type_summary = df_clean.groupby(['source', 'type']).size().unstack(fill_value=0)
    print("\nSource × Type:")
    print(type_summary)
    
    # Price stats
    price_stats = df_clean.groupby('type')['price_num'].agg([
        'count', 'mean', 'median', 'std', 'min', 'max'
    ]).round(0)
    print("\nPrices (TND):")
    print(price_stats)
    
    # Locations
    if 'location_clean' in df_clean.columns:
        top_locs = df_clean['location_clean'].value_counts().head(10)
    else:
        top_locs = pd.Series()
    print("\nTop Locations:")
    print(top_locs)
    
    return type_summary, price_stats, top_locs

type_summary, price_stats, top_locs = eda_stats(df_clean)

# =============================================================================
# 4. VISUALS (Safe)
# =============================================================================
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle(f'Estate-Mind Tunisia: {len(df_clean):,} Valid Listings', fontsize=16)

# Plot 1: Price by type
if len(df_clean) > 0:
    bp = sns.boxplot(data=df_clean, x='type', y='price_num', ax=axes[0,0])
    bp.set_yscale('log')
    axes[0,0].set_title('Price by Type (log TND)')

# Plot 2: Sources
if 'source' in df_clean.columns and len(df_clean['source'].unique()) > 0:
    source_counts = df_clean['source'].value_counts()
    source_counts.plot(kind='bar', ax=axes[0,1])
    axes[0,1].set_title('Listings by Source')
    axes[0,1].tick_params(axis='x', rotation=45)

# Plot 3: Locations
if len(top_locs) > 0:
    top_locs.head(8).plot(kind='barh', ax=axes[0,2])
    axes[0,2].set_title('Top Locations')

# Plot 4: Rent vs bedrooms
rent_data = df_clean[(df_clean['type'] == 'rent') & df_clean['bedrooms'].notna()]
if len(rent_data) > 2:
    sns.boxplot(data=rent_data, x='bedrooms', y='price_num', ax=axes[1,0])
    axes[1,0].set_title('Rent Price vs Bedrooms')

# Plot 5: Size vs price (sales)
sale_data = df_clean[(df_clean['type'] == 'sale') & df_clean['size_m2'].notna()]
if len(sale_data) > 2:
    sns.scatterplot(data=sale_data, x='size_m2', y='price_num', hue='source', ax=axes[1,1])
    axes[1,1].set_yscale('log')
    axes[1,1].set_title('Sale: Size vs Price')

# Plot 6: Type pie
type_pie = df_clean['type'].value_counts()
if len(type_pie) > 0:
    type_pie.plot(kind='pie', ax=axes[1,2], autopct='%1.1f%%')
    axes[1,2].set_title('Rent vs Sale')

plt.tight_layout()
plt.savefig('output/visuals/eda_complete.png', dpi=300, bbox_inches='tight')
plt.show()
print("✅ Visuals: output/visuals/eda_complete.png")

# =============================================================================
# 5. EXPORTS & REPORT
# =============================================================================
# Subsets
df_clean[df_clean['type'] == 'rent'].to_csv('output/rentals_clean.csv', index=False)
df_clean[df_clean['type'] == 'sale'].to_csv('output/sales_clean.csv', index=False)

# Report
report = f"""
ESTATE-MIND PHASE 2 - DATA UNDERSTANDING
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}

📊 DATA SUMMARY:
Total Raw: {len(df)} listings
Total Clean: {len(df_clean)} valid listings ({len(df_clean)/len(df)*100:.1f}%)

{type_summary.to_string()}

PRICES:
{price_stats.to_string()}

TOP LOCATIONS:
{top_locs.to_string()}

✅ FILES READY FOR SUBMISSION:
• output/estate_data_cleaned.csv
• output/visuals/eda_complete.png ← PLOT FOR PROF
• output/phase2_report.txt ← COPY TO WORD

🔜 Phase 3: ML price prediction
"""
with open('output/phase2_report.txt', 'w', encoding='utf-8') as f:
    f.write(report)

print("\n🎉 PIPELINE 100% COMPLETE!")
print(report)
