import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import os

st.set_page_config(page_title="Tunisia Real Estate Predictor", layout="wide")

st.title("🏠 Tunisia Real Estate Price Predictor")
st.markdown("**4,712 listings | RF MAE: 69k TND | is_sale 71% importance**")

# Load data
@st.cache_data
def load_model_data():
    df = pd.read_csv('data\\raw\\tunisia_estate_complete_clean.csv')
    return df

df = load_model_data()
st.success(f"✅ Loaded {len(df):,} production listings")

# Optional: Safe image load
if os.path.exists('final/complete_eda.jpg'):
    st.image('final/complete_eda.jpg', caption="EDA Insights", use_column_width=True)
else:
    st.info("📊 EDA plots available in final/complete_eda.jpg")

# Sidebar inputs
st.sidebar.header("Property Details")
size = st.sidebar.slider("Size (m²)", 30, 1000, 120)
rooms = st.sidebar.slider("Rooms", 1, 10, 3)
category = st.sidebar.selectbox("Category", df['category'].unique())

# Predict
if st.sidebar.button("🚀 Predict Price", type="primary"):
    is_sale = 1 if 'sale' in str(category).lower() else 0
    X = pd.DataFrame({
        'size_numeric': [size], 
        'rooms_numeric': [rooms], 
        'is_sale': [is_sale]
    })
    
    # Quick retrain
    df_ml = df[['size_numeric', 'rooms_numeric', 'is_sale', 'price_numeric']].dropna()
    X_train, X_test, y_train, y_test = train_test_split(
        df_ml.drop('price_numeric', axis=1), df_ml['price_numeric'], test_size=0.2, random_state=42
    )
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    pred = model.predict(X)[0]
    st.metric("**Predicted Price**", f"{pred:,.0f} TND", f"{pred/1000:.0f}k TND")
    
    # Confidence
    st.write(f"**Model Score**: Train MAE {mean_absolute_error(y_train, model.predict(X_train)):.0f} | Test {mean_absolute_error(y_test, model.predict(X_test)):.0f}")
    
    # Similar listings
    similars = df[(df['category']==category) & 
                  (df['size_numeric'].between(size*0.8, size*1.2))]
    st.subheader("🔍 Similar Listings")
    st.dataframe(similars[['title', 'price', 'size_m2', 'rooms']].head(10), use_container_width=True)

# Live stats
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Listings", len(df))
col2.metric("Avg Price", f"{df['price_numeric'].mean():,.0f} TND")
col3.metric("Median Size", f"{df['size_numeric'].median():.0f} m²")
col4.metric("Avg Rooms", f"{df['rooms_numeric'].mean():.1f}")

st.markdown("---")
st.caption("ESPRIT DS Project 2025/26 - Phase 3 Deployment Ready")
