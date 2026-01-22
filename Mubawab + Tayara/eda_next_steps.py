import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('tunisia_real_estate_combined_clean.csv')

# Rentals (low price proxy <10k TND)
rentals = df[df['price_clean'] < 10000].copy()
print("Rentals overview:\n", rentals['source'].value_counts())
print("Avg rental price by top cities:\n", rentals.groupby('location_clean')['price_clean'].mean().sort_values(ascending=False).head())

# Save filtered
rentals.to_csv('tunisia_rentals_filtered.csv', index=False)
print("Saved filtered rentals!")
