import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load model, scaler, and encoders
model = joblib.load("tree_model.pkl")
scaler = joblib.load("scaler.pkl")
target_encodings = joblib.load("target_encoding.pkl")

st.title("🚗 Cars24 - Car Price Predictor")
st.write("Predict the resale value of your car based on features.")

# User inputs
make = st.selectbox("Car Make", list(target_encodings['make'].keys()))
model_name = st.selectbox("Car Model", list(target_encodings['model'].keys()))
year = st.number_input("Year of Manufacture", min_value=1995, max_value=2025, value=2015)
km_driven = st.number_input("Kilometers Driven", min_value=0, value=50000)
mileage = st.number_input("Mileage (kmpl)", min_value=5.0, max_value=50.0, value=18.0)
engine = st.number_input("Engine Capacity (cc)", min_value=600, max_value=5000, value=1200)
max_power = st.number_input("Max Power (bhp)", min_value=20.0, max_value=500.0, value=80.0)
seats = st.selectbox("Seating Capacity", [2, 4, 5, 6, 7])
owner = st.selectbox("Previous Owners", [0, 1, 2, 3])
fuel_type = st.selectbox("Fuel Type", ['Diesel', 'Petrol', 'CNG', 'Electric'])
transmission = st.selectbox("Transmission", ['Manual', 'Automatic'])
seller_type = st.selectbox("Seller Type", ['Dealer', 'Individual', 'Trustmark Dealer'])

if st.button("Predict Price"):
    try:
        # Feature engineering
        
        mileage_per_engine = mileage / engine

        # Target encoding
        make_encoded = target_encodings['make'].get(make, 0)
        model_encoded = target_encodings['model'].get(model_name, 0)

        # One-hot style manual columns
        fuel_dict = {'Diesel': 0, 'Petrol': 0, 'CNG': 0, 'Electric': 0}
        fuel_dict[fuel_type] = 1

        seller_dict = {'Dealer': 0, 'Individual': 0, 'Trustmark Dealer': 0}
        seller_dict[seller_type] = 1

        manual = 1 if transmission == 'Manual' else 0
        seat_5 = 1 if seats == 5 else 0
        more_than_5 = 1 if seats > 5 else 0

        # Construct feature vector
        input_data = pd.DataFrame([{
            'make': make_encoded,
            'model': model_encoded,
            'year': year,
            'km_driven': km_driven,
            'mileage': mileage,
            'engine': engine,
            'max_power': max_power,
            
            'mileage_per_engine': mileage_per_engine,
            'Diesel': fuel_dict['Diesel'],
            'Petrol': fuel_dict['Petrol'],
            'Electric': fuel_dict['Electric'],
            'LPG': 0,  # default unless needed
            'Manual': manual,
            'Trustmark Dealer': seller_dict['Trustmark Dealer'],
            'Individual': seller_dict['Individual'],
            '5': seat_5,
            '>5': more_than_5
        }])

        # Scale the input
        scaled_input = scaler.transform(input_data)

        # Predict
        prediction = model.predict(scaled_input)[0]
        st.success(f"Estimated Resale Price: ₹ {prediction:,.2f}")

    except Exception as e:
        st.error(f"Error: {e}")
