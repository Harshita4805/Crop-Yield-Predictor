import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Load dataset and model
df = pd.read_csv("crop_production.csv")
model = pickle.load(open("model.pkl", "rb"))

# Clean data exactly like training
df.dropna(inplace=True)
df = df[df['Production'] > 0]
df = df[df['Area'] > 0]

# Create yield per hectare column
df['Yield_per_hectare'] = (df['Production'] / df['Area']) * 1000
df = df[df['Yield_per_hectare'] < df['Yield_per_hectare'].quantile(0.99)]
df = df[df['Yield_per_hectare'] > df['Yield_per_hectare'].quantile(0.01)]

# Get unique values for dropdowns
states  = sorted(df['State_Name'].unique())
crops   = sorted(df['Crop'].unique())
seasons = sorted(df['Season'].unique())

# Fit label encoders exactly like training
le_state    = LabelEncoder().fit(df['State_Name'])
le_district = LabelEncoder().fit(df['District_Name'])
le_crop     = LabelEncoder().fit(df['Crop'])
le_season   = LabelEncoder().fit(df['Season'])

st.set_page_config(page_title="Crop Yield Predictor", page_icon="🌾")
st.title("🌾 Crop Yield Prediction")
st.markdown("Enter field details below to predict expected crop yield.")

# Input form
col1, col2 = st.columns(2)

with col1:
    selected_state    = st.selectbox("State", states)
    districts         = sorted(df[df['State_Name'] == selected_state]['District_Name'].unique())
    selected_district = st.selectbox("District", districts)
    selected_crop     = st.selectbox("Crop type", crops)

with col2:
    selected_season = st.selectbox("Season", seasons)
    area            = st.number_input("Area (hectares)", min_value=1.0, value=100.0)
    year            = st.number_input("Year", min_value=2000, max_value=2030, value=2023)

# Predict button
if st.button("Predict Yield", use_container_width=True):

    # Encode inputs exactly like training
    state_enc    = le_state.transform([selected_state])[0]
    district_enc = le_district.transform([selected_district])[0]
    crop_enc     = le_crop.transform([selected_crop])[0]
    season_enc   = le_season.transform([selected_season])[0]

    features   = np.array([[state_enc, district_enc, crop_enc, year, season_enc, area]])
    prediction = model.predict(features)[0]

    st.success(f"Predicted Yield: **{prediction:,.0f} kg/hectare**")

    # Comparison chart using Yield_per_hectare
    filtered = df[
        (df['Crop'] == selected_crop) &
        (df['State_Name'] == selected_state) &
        (df['District_Name'] == selected_district)
    ]['Yield_per_hectare']

    if len(filtered) > 0:
        avg_yield = filtered.mean()
        max_yield = filtered.max()

        fig, ax = plt.subplots(figsize=(6, 3))
        bars = ax.barh(
            [
                f"Max yield of {selected_crop} in {selected_district}",
                f"Avg yield of {selected_crop} in {selected_district}",
                "Your prediction"
            ],
            [max_yield, avg_yield, prediction],
            color=["#e07b54", "#f0c05a", "#3a9e64"]
        )
        ax.set_xlabel("kg / hectare")
        ax.set_title(f"{selected_crop} yield — {selected_district}, {selected_state}")

        # Add value labels on bars
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 5, bar.get_y() + bar.get_height()/2,
                    f'{width:,.0f}', va='center', fontsize=9)

        plt.tight_layout()
        st.pyplot(fig)

    else:
        st.info(f"No historical data for {selected_crop} in {selected_state}.")

    # Show selected inputs summary
    st.markdown("---")
    st.markdown(f"**State:** {selected_state} | **District:** {selected_district} | **Crop:** {selected_crop} | **Season:** {selected_season}")

st.markdown("---")
st.caption("Note: This is an ML prediction model. Results may vary based on actual conditions.")