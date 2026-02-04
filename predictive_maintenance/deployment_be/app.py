
import os
import joblib
import pandas as pd
import logging
from flask import Flask, request, jsonify
from huggingface_hub import hf_hub_download

# -------------------------------------------------------------------
# Logging
# -------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# Flask App Initialization
# -------------------------------------------------------------------
app = Flask("Predictive Maintenance Predictor")

# -------------------------------------------------------------------
# Hugging Face Authentication
# -------------------------------------------------------------------
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN missing in Hugging Face Space secrets")

# -------------------------------------------------------------------
# Load Model
# -------------------------------------------------------------------
REPO_ID = "samdurai102024/predictive-maintenance-be"
FILENAME = "production/model.joblib"

logger.info("Downloading model from Hugging Face Hub...")
model_path = hf_hub_download(
    repo_id=REPO_ID,
    filename=FILENAME,
    token=HF_TOKEN
)

model = joblib.load(model_path)
logger.info("Model loaded successfully")

# -------------------------------------------------------------------
# Feature Engineering Functions
# -------------------------------------------------------------------
def rpm_category_fn(rpm):
    if rpm <= 742:
        return "High Risk"
    elif rpm <= 886:
        return "Moderate Risk"
    else:
        return "Lower Risk"

def oil_pressure_category_fn(op):
    if op <= 2.5:
        return "Low Risk"
    elif op <= 5.0:
        return "Mid Risk"
    else:
        return "High Risk"

def fuel_pressure_category_fn(fp):
    if fp <= 4.0:
        return "Low Risk"
    elif fp <= 5.1:
        return "Mid Risk"
    elif fp <= 6.2:
        return "High Risk"
    else:
        return "Very High Risk"

def coolant_pressure_category_fn(cp):
    if cp <= 3.1:
        return "Normal-to-Elevated"
    elif cp <= 3.8:
        return "Critical Pressure"
    else:
        return "Relief / Saturation Regime"

def fuel_oil_risk_fn(fp, ot):
    if (fp > 5.1) and (ot < 77.2):
        return "High Risk"
    elif (fp > 5.1) and (ot >= 77.2):
        return "Mid Risk"
    elif (fp <= 5.1) and (ot < 77.2):
        return "Mid Risk"
    else:
        return "Low Risk"

def coolant_temp_category_fn(ct):
    if ct < 69.17:
        return "High"
    elif ct < 72.27:
        return "Elevated"
    elif ct < 80.55:
        return "Moderate"
    else:
        return "Lower"

# -------------------------------------------------------------------
# Schema
# -------------------------------------------------------------------
RAW_COLUMNS = [
    "Engine_rpm",
    "Lub_oil_pressure",
    "Fuel_pressure",
    "Coolant_pressure",
    "lub_oil_temp",
    "Coolant_temp"
]

FEATURE_COLUMNS = RAW_COLUMNS + [
    "rpm_category",
    "oil_pressure_category",
    "fuel_pressure_category",
    "coolant_pressure_category",
    "fuel_pressure_oil_temp",
    "coolant_temp_category"
]

# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace spaces with underscores in column names
    """
    rename_mapping = {col: col.replace(" ", "_") for col in df.columns}
    return df.rename(columns = rename_mapping)

def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["rpm_category"] = df["Engine_rpm"].apply(rpm_category_fn)
    df["oil_pressure_category"] = df["Lub_oil_pressure"].apply(oil_pressure_category_fn)
    df["fuel_pressure_category"] = df["Fuel_pressure"].apply(fuel_pressure_category_fn)
    df["coolant_pressure_category"] = df["Coolant_pressure"].apply(coolant_pressure_category_fn)
    df["fuel_pressure_oil_temp"] = df.apply(
        lambda r: fuel_oil_risk_fn(r["Fuel_pressure"], r["lub_oil_temp"]), axis=1
    )
    df["coolant_temp_category"] = df["Coolant_temp"].apply(coolant_temp_category_fn)

    return df

# -------------------------------------------------------------------
# Health
# -------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}, 200

# -------------------------------------------------------------------
# Home
# -------------------------------------------------------------------
@app.get("/")
def home():
    return {"service": "Predictive Maintenance Backend", "status": "running"}

# -------------------------------------------------------------------
# Single Prediction (ROW-ALIGNED)
# -------------------------------------------------------------------
@app.post("/v1/maintenance")
def predict_single():
    try:
        payload = request.get_json(silent=True)
        if not payload:
            return jsonify({"error": "Invalid JSON payload"}), 400

        df = pd.DataFrame([payload])
        df = normalize_columns(df)

        missing = [c for c in RAW_COLUMNS if c not in df.columns]
        if missing:
            return jsonify({"error": "Missing fields", "missing_fields": missing}), 400

        df = add_engineered_features(df)
        df_model = df[FEATURE_COLUMNS].astype(float, errors="ignore")

        prediction = int(model.predict(df_model)[0])

        confidence = None
        if hasattr(model, "predict_proba"):
            confidence = float(model.predict_proba(df_model)[0].max())

        response = {
            "features": df[RAW_COLUMNS].iloc[0].to_dict(),
            "Engine_Condition": prediction,
            "confidence": confidence
        }

        return jsonify(response), 200

    except Exception as e:
        logger.exception("Single prediction failed")
        return jsonify({"error": "Single prediction failed", "details": str(e)}), 500

# -------------------------------------------------------------------
# Batch Prediction (ROW-ALIGNED)
# -------------------------------------------------------------------
@app.post("/v1/maintenance/batch")
def predict_batch():
    try:
        if "file" not in request.files:
            return jsonify({"error": "CSV file missing"}), 400

        df = pd.read_csv(request.files["file"])
        df = normalize_columns(df)

        missing = [c for c in RAW_COLUMNS if c not in df.columns]
        if missing:
            return jsonify({"error": "Missing columns", "missing_columns": missing}), 400

        df = add_engineered_features(df)
        df_model = df[FEATURE_COLUMNS].astype(float, errors="ignore")

        predictions = model.predict(df_model)

        results = []
        for i, pred in enumerate(predictions):
            row = df.iloc[i][RAW_COLUMNS].to_dict()
            row["Engine_Condition"] = int(pred)
            results.append(row)

        return jsonify({"results": results}), 200

    except Exception as e:
        logger.exception("Batch prediction failed")
        return jsonify({"error": "Batch prediction failed", "details": str(e)}), 500

# -------------------------------------------------------------------
# Local Run
# -------------------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)
