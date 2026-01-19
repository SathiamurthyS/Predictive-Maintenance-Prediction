
import os
import joblib
import pandas as pd
import traceback
from flask import Flask, request, jsonify
from huggingface_hub import hf_hub_download, HfApi

# -------------------------------------------------------------------
# Flask App Initialization
# -------------------------------------------------------------------
app = Flask("Predictive Maintenance Predictor")

# -------------------------------------------------------------------
# Hugging Face Authentication (PRODUCTION SAFE)
# -------------------------------------------------------------------
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN missing in Hugging Face Space secrets")

api = HfApi(token=HF_TOKEN)

# -------------------------------------------------------------------
# Load Model from Hugging Face Hub
# -------------------------------------------------------------------
REPO_ID = "samdurai102024/predictive-maintenance-be"
FILENAME = "best_predictive_maintenance_prediction_v1.joblib"

model_path = hf_hub_download(
    repo_id=REPO_ID,
    filename=FILENAME,
    token=HF_TOKEN
)

model = joblib.load(model_path)

print(" Model loaded successfully from Hugging Face")
print(model)

# -------------------------------------------------------------------
# Health Check 
# -------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}, 200

# -------------------------------------------------------------------
# Home
# -------------------------------------------------------------------
@app.get("/")
def home():
    return "Predictive Maintenance Backend is running"

# -------------------------------------------------------------------
# Single Prediction Endpoint
# -------------------------------------------------------------------
@app.post("/v1/maintenance")
def predict_maintenance():
    try:
        payload = request.get_json(silent=True)

        if payload is None:
            return jsonify({
                "error": "Invalid or missing JSON payload"
            }), 400

        required_fields = [
            "Engine_rpm",
            "Lub_oil_pressure",
            "Fuel_pressure",
            "Coolant_pressure",
            "lub_oil_temp",
            "Coolant_temp",
            "rpm_category",
            "oil_pressure_category"
        ]

        missing = [f for f in required_fields if f not in payload]
        if missing:
            return jsonify({
                "error": "Missing required fields",
                "missing_fields": missing
            }), 400

        # Convert payload to DataFrame
        input_df = pd.DataFrame([payload])

        prediction = model.predict(input_df)

        return jsonify({
            "prediction": int(prediction[0])
        })

    except Exception as e:
        app.logger.exception("Prediction failed")
        return jsonify({
            "error": "Internal server error",
            "details": str(e)
        }), 500

# -------------------------------------------------------------------
# Batch Prediction Endpoint
# -------------------------------------------------------------------
@app.post("/v1/maintenancebatch")
def predict_maintenance_batch():
    try:
        if "file" not in request.files:
            return {"error": "CSV file missing"}, 400

        file = request.files["file"]
        input_df = pd.read_csv(file)

        if "Engine_Id" not in input_df.columns:
            return {"error": "Engine_Id column missing"}, 400

        features_df = input_df.drop(columns=["Engine_Id"])
        predictions = model.predict(features_df).tolist()

        output = dict(zip(input_df["Engine_Id"].tolist(), predictions))

        return jsonify(output)

    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}, 500
