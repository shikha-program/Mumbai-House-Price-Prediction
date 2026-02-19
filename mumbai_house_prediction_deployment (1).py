# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -----------------------------
# Load Model & Encoders
# -----------------------------
model = joblib.load("model.pkl")
encoders = joblib.load("encoder.pkl")

st.set_page_config(page_title="Mumbai House Price Prediction", page_icon="🏠")

st.title("🏠 Mumbai House Price Prediction App")
st.write("Enter property details below:")

# -----------------------------
# User Inputs
# -----------------------------

age = st.number_input("Age of Property", min_value=0, max_value=100)

city = st.selectbox("City", encoders["City"].classes_)
area = st.selectbox("Area", encoders["Area"].classes_)
property_type = st.selectbox("Property Type", encoders["property_type"].classes_)

years_of_exp = st.number_input("Years of Experience", min_value=0, max_value=50)

# -----------------------------
# Prediction Button
# -----------------------------

if st.button("Predict Price"):

    # Create DataFrame
    df = pd.DataFrame({
        "Age": [age],
        "City": [city],
        "Area": [area],
        "property_type": [property_type],
        "Years of Experience": [years_of_exp]
    })

    # -----------------------------
    # Apply Encoding (Same as Training)
    # -----------------------------
    for col in ["City", "Area", "property_type"]:
        df[col] = encoders[col].transform(df[col])

    # -----------------------------
    # If you applied log transform in training,
    # apply same here (example)
    # df["Age"] = np.log1p(df["Age"])
    # -----------------------------

    # Ensure correct column order (VERY IMPORTANT)
    df = df[model.feature_names_in_]

    # -----------------------------
    # Make Prediction
    # -----------------------------
    prediction = model.predict(df)

    st.success(f"💰 Predicted House Price: ₹ {prediction[0]:,.2f}")
