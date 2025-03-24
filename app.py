import os
os.system("pip install --no-cache-dir joblib")
import joblib
import streamlit as st
import numpy as np
import pandas as pd


# Load trained model and encoders
model = joblib.load("model/car_price_model.pkl")
ohe = joblib.load("model/encoder.pkl")
scaler = joblib.load("model/scaler.pkl")

# Load dataset to extract unique values
df = pd.read_csv("dataset/train-data.csv")  # Ensure correct path
df['Brand'] = df['Name'].str.split().str[0]  # Extract brand from Name
unique_brands = sorted(df['Brand'].unique())  # Get unique brands
unique_locations = sorted(df['Location'].unique())  # Get unique locations

# Set page configuration
st.set_page_config(page_title="Car Price Prediction 🚗", page_icon="🚗", layout="wide")

# Custom CSS for Styling
st.markdown(
    """
    <style>
        .title { text-align: center; font-size: 40px; font-weight: bold; color: #FF4B4B; }
        .subtitle { text-align: center; font-size: 18px; color: #4B4B4B; margin-bottom: 20px; }
        .stButton > button { background-color: #FF4B4B; color: white; font-size: 18px; padding: 10px 15px; border-radius: 10px; width: 100%; }
        .stButton > button:hover { background-color: #ff1f1f; }
        .stSuccess { font-size: 24px; font-weight: bold; color: #28A745; text-align: center; }
    </style>
    """,
    unsafe_allow_html=True
)

# Main Title
st.markdown('<h1 class="title">🚗 Car Price Prediction System 💰</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Enter car details to get an estimated price.</p>', unsafe_allow_html=True)

# Sidebar for input
with st.sidebar:
    st.markdown("## 🔹 Enter Car Details")

    # User Input Fields
    brand = st.selectbox("🚘 Car Brand", unique_brands)
    fuel_type = st.selectbox("⛽ Fuel Type", ['Petrol', 'Diesel'])
    transmission = st.selectbox("🔀 Transmission", ['Manual', 'Automatic'])
    owner_type = st.selectbox("👤 Owner Type", ['Second', 'Third'])
    location = st.selectbox("📍 Location", unique_locations)  # Now dynamically loaded from dataset
    manufacturing_year = st.number_input("📅 Manufacturing Year", min_value=1990, value=2015, step=1)
    kms_driven = st.number_input("🚗 Kilometers Driven", min_value=0, value=50000, step=1000)
    mileage = st.number_input("⛽ Mileage (kmpl)", min_value=0.0, value=20.0, step=0.1)
    engine = st.number_input("⚙ Engine Capacity (CC)", min_value=500, value=1500, step=100)
    power = st.number_input("🏎 Power (BHP)", min_value=20, value=100, step=5)
    seats = st.selectbox("🛋 Seats", [2, 4, 5, 6])

# Compute car age
car_age = 2025 - manufacturing_year

# Prepare input data
input_data = pd.DataFrame({
    'Kilometers_Driven': [kms_driven],
    'Mileage': [mileage],
    'Engine': [engine],
    'Power': [power],
    'Seats': [seats],
    'Car_Age': [car_age],
    'Fuel_Type': [fuel_type],
    'Transmission': [transmission],
    'Owner_Type': [owner_type],
    'Location': [location],
    'Brand': [brand]
})

# Encode categorical features
encoded_input = ohe.transform(input_data[['Fuel_Type', 'Transmission', 'Owner_Type', 'Location', 'Brand']])
encoded_df = pd.DataFrame(encoded_input, columns=ohe.get_feature_names_out())

# Scale numerical features
scaled_numerical = scaler.transform(input_data[['Kilometers_Driven', 'Mileage', 'Engine', 'Power', 'Seats', 'Car_Age']])
scaled_df = pd.DataFrame(scaled_numerical, columns=['Kilometers_Driven', 'Mileage', 'Engine', 'Power', 'Seats', 'Car_Age'])

# Combine features
final_input = pd.concat([scaled_df, encoded_df], axis=1)

# Prediction Button in Center
st.markdown("<br>", unsafe_allow_html=True)
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if st.button("💰 Estimate Car Price"):
        predicted_price = model.predict(final_input)
        estimated_price = np.expm1(predicted_price)[0]  # Convert log-transformed prediction back
        
        st.markdown(f'<p class="stSuccess">💰 Estimated Car Price: ₹{estimated_price:.2f} Lakhs</p>', unsafe_allow_html=True)
