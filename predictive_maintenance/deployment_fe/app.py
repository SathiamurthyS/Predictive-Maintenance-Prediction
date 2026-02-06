import os
import logging
import streamlit as st
import pandas as pd
import requests
import io

# -----------------------------
# Suppress Streamlit Warnings
# -----------------------------
os.environ["STREAMLIT_SERVER_HEADLESS"] = "1"
logging.getLogger("streamlit").setLevel(logging.ERROR)

# -----------------------------
# App Config
# -----------------------------
st.set_page_config(
    page_title="Predictive Maintenance Predictor",
    layout="centered"
)

st.title("Predictive Maintenance Predictor")
st.write(
    "This app predicts **Predictive Maintenance** based on engine sensor data.\n\n"
    "Developed and Deployed by **Sathiamurthy Samidurai (AIML Student)**."
)

# -----------------------------
# Backend URLs
# -----------------------------
BACKEND_URL_SINGLE = "https://samdurai102024-predictive-maintenance-be.hf.space/v1/maintenance"
BACKEND_URL_BATCH = "https://samdurai102024-predictive-maintenance-be.hf.space/v1/maintenance/batch"

# -----------------------------
# Backend Sanity Check
# -----------------------------
try:
    health_resp = requests.get(
        BACKEND_URL_SINGLE.replace("/v1/maintenance", "/health"),
        timeout=5
    )
    if health_resp.status_code == 200:
        st.info(" Backend is reachable and healthy")
    else:
        st.warning(f"⚠ Backend reachable but returned {health_resp.status_code}")
except requests.exceptions.RequestException:
    st.error(" Backend service is not reachable. Predictions will fail!")

# -----------------------------
# User Inputs (Single)
# -----------------------------
st.subheader("Single Engine Prediction")

Engine_rpm = st.number_input("Engine RPM", min_value=0.0, step=10.0)
Lub_oil_pressure = st.number_input("Lub Oil Pressure", min_value=0.0)
Fuel_pressure = st.number_input("Fuel Pressure", min_value=0.0)
Coolant_pressure = st.number_input("Coolant Pressure", min_value=0.0)
lub_oil_temp = st.number_input("Lub Oil Temperature", min_value=0.0)
Coolant_temp = st.number_input("Coolant Temperature", min_value=0.0)

payload = {
    "Engine_rpm": Engine_rpm,
    "Lub_oil_pressure": Lub_oil_pressure,
    "Fuel_pressure": Fuel_pressure,
    "Coolant_pressure": Coolant_pressure,
    "lub_oil_temp": lub_oil_temp,
    "Coolant_temp": Coolant_temp,
}

# -----------------------------
# Single Prediction
# -----------------------------
if st.button("Predict", type="primary"):
    try:
        st.info(" Sending request to backend...")
        response = requests.post(BACKEND_URL_SINGLE, json=payload, timeout=10)

        if response.status_code == 200:
            result = response.json()

            # Validate expected keys
            required_keys = {"features", "Engine_Condition"}
            if not required_keys.issubset(result):
                st.error("Unexpected response format from backend")
                st.json(result)
                st.stop()

            # Build aligned row
            row = result["features"].copy()
            row["Engine_Condition"] = result["Engine_Condition"]

            if "confidence" in result:
                row["confidence"] = result["confidence"]

            aligned_df = pd.DataFrame([row])

            st.success(" Prediction Successful")

            st.metric(
                "Engine Condition",
                result["Engine_Condition"]
            )

            if "confidence" in result:
                st.write(f"**Confidence:** {result['confidence']:.2f}")

            st.subheader("Prediction Details (Aligned with Features)")
            st.dataframe(aligned_df)

        else:
            st.error(f"API Error {response.status_code}")
            st.write(response.text)

    except requests.exceptions.RequestException as e:
        st.error(" Backend service unavailable")
        st.write(str(e))


# -----------------------------
# Batch Prediction
# -----------------------------
st.divider()
st.subheader("Batch Prediction")

uploaded_file = st.file_uploader(
    "Upload CSV file",
    type=["csv"],
    help="CSV must contain the same raw feature columns used during training"
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.dataframe(df.head())
    st.info(f" File loaded with {len(df)} rows")

    if st.button("Predict for Batch", type="primary"):
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)

        try:
            st.info(" Sending batch request to backend...")
            response = requests.post(
                BACKEND_URL_BATCH,
                files={"file": ("batch.csv", csv_buffer.getvalue())},
                timeout=30
            )

            if response.status_code == 200:
                api_response = response.json()

                if "results" not in api_response:
                    st.error("Unexpected batch response format")
                    st.json(api_response)
                    st.stop()

                results_df = pd.DataFrame(api_response["results"])

                st.success(" Batch Prediction Successful")
                st.subheader("Batch Prediction Results (Aligned)")
                st.dataframe(results_df)

                # Optional CSV download
                csv_out = results_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇ Download Predictions as CSV",
                    csv_out,
                    file_name="maintenance_batch_predictions.csv",
                    mime="text/csv"
                )

            else:
                st.error(f"⚠ API Error {response.status_code}")
                st.write(response.text)

        except requests.exceptions.RequestException as e:
            st.error(" Backend service unavailable")
            st.write(str(e))
