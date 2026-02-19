# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="Mumbai House Price Prediction", page_icon="🏠")

st.title("🏠 Mumbai House Price Prediction App")
st.write("Enter property details below:")

# -------------------------------------------------
# STEP 1: Create Dummy Training Data
# (Replace with your real dataset if you have CSV)
# -------------------------------------------------

data = pd.DataFrame({
    "Age": [5, 10, 2, 15, 7, 3],
    "City": ["Mumbai", "Mumbai", "Pune", "Mumbai", "Pune", "Mumbai"],
    "Area": ["Andheri", "Bandra", "Wakad", "Dadar", "Hinjewadi", "Borivali"],
    "property_type": ["Flat", "Flat", "Villa", "Flat", "Villa", "Flat"],
    "Years of Experience": [2, 5, 1, 10, 3, 4],
    "Price": [12000000, 15000000, 8000000, 20000000, 9000000, 13000000]
})

# -------------------------------------------------
# STEP 2: Encode Categorical Columns
# -------------------------------------------------

encoders = {}
for col in ["City", "Area", "property_type"]:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    encoders[col] = le

# -------------------------------------------------
# STEP 3: Train Model
# -------------------------------------------------

X = data.drop("Price", axis=1)
y = data["Price"]

model = RandomForestRegressor()
model.fit(X, y)

# -------------------------------------------------
# USER INPUTS
# -------------------------------------------------

age = st.number_input("Age of Property", min_value=0, max_value=100)

city = st.selectbox("City", encoders["City"].classes_)
area = st.selectbox("Area", encoders["Area"].classes_)
property_type = st.selectbox("Property Type", encoders["property_type"].classes_)

years_of_exp = st.number_input("Years of Experience", min_value=0, max_value=50)

# -------------------------------------------------
# PREDICTION
# -------------------------------------------------

if st.button("Predict Price"):

    df = pd.DataFrame({
        "Age": [age],
        "City": [city],
        "Area": [area],
        "property_type": [property_type],
        "Years of Experience": [years_of_exp]
    })

    # Encode input
    for col in ["City", "Area", "property_type"]:
        df[col] = encoders[col].transform(df[col])

    prediction = model.predict(df)

    st.success(f"💰 Predicted House Price: ₹ {prediction[0]:,.2f}")
