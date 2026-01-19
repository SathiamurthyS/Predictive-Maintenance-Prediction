import streamlit as st
import pandas as pd
import requests

# -----------------------------
# App Config
# -----------------------------
st.set_page_config(
    page_title="Predictive Maintenance Predictor",
    layout="centered"
)

st.title("Predictive Maintenance Predictor")
st.write(
    "This app predicts **Predictive Maintenance** based on engine sensor data.  \n"
    "Developed and Deployed by **Sathiamurthy Samidurai (AIML Student)**."
)

# -----------------------------
# Backend URLs
# -----------------------------
BACKEND_URL = "https://samdurai102024-predictive-maintenance-be.hf.space/v1/maintenance"
BACKEND_URL_BATCH = "https://samdurai102024-predictive-maintenance-be.hf.space/v1/maintenancebatch"

# -----------------------------
# User Inputs
# -----------------------------
st.subheader("Single Engine Prediction")

Engine_rpm = st.number_input("Engine RPM", min_value=0.0, step=10.0)
Lub_oil_pressure = st.number_input("Lub Oil Pressure", min_value=0.0)
Fuel_pressure = st.number_input("Fuel Pressure", min_value=0.0)
Coolant_pressure = st.number_input("Coolant Pressure", min_value=0.0)
lub_oil_temp = st.number_input("Lub Oil Temperature", min_value=0.0)
Coolant_temp = st.number_input("Coolant Temperature", min_value=0.0)

rpm_category = st.selectbox(
    "RPM Risk Category",
    ["Lower Risk", "Moderate Risk", "High Risk"]
)

oil_pressure_category = st.selectbox(
    "Oil Pressure Risk Category",
    ["Low Risk", "Mid Risk", "High Risk"]
)

# -----------------------------
# Build Payload (MATCHES FLASK)
# -----------------------------
payload = {
    "Engine_rpm": Engine_rpm,
    "Lub_oil_pressure": Lub_oil_pressure,
    "Fuel_pressure": Fuel_pressure,
    "Coolant_pressure": Coolant_pressure,
    "lub_oil_temp": lub_oil_temp,
    "Coolant_temp": Coolant_temp,
    "rpm_category": rpm_category,
    "oil_pressure_category": oil_pressure_category
}

# -----------------------------
# Single Prediction
# -----------------------------
if st.button("Predict", type="primary"):
    try:
        response = requests.post(
            BACKEND_URL,
            json=payload,
            timeout=10
        )

        if response.status_code == 200:
            result = response.json()
            st.success("Prediction Successful")
            st.metric(
                label="Predictive Maintenance Result",
                value=result["prediction"]
            )
        else:
            st.error(f"API Error {response.status_code}")
            st.json(response.json())

    except requests.exceptions.RequestException as e:
        st.error("Backend service unavailable")
        st.write(str(e))

# -----------------------------
# Batch Prediction
# -----------------------------
st.divider()
st.subheader("Batch Prediction")

file = st.file_uploader(
    "Upload CSV file",
    type=["csv"],
    help="CSV must contain the same feature columns used during training"
)

if file is not None:
    if st.button("Predict for Batch", type="primary"):
        try:
            response = requests.post(
                BACKEND_URL_BATCH,
                files={"file": file},
                timeout=20
            )

            if response.status_code == 200:
                st.success("Batch Prediction Successful")
                st.json(response.json())
            else:
                st.error(f"API Error {response.status_code}")
                st.write(response.text)

        except requests.exceptions.RequestException as e:
            st.error("Backend service unavailable")
            st.write(str(e))
