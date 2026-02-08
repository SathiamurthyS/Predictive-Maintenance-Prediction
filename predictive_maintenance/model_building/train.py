
# ============================================================
# Final Training / Registration Script (PERFORMANCE SAFE)
# ============================================================

import os
import time
import subprocess
import json
import joblib
import warnings
import numpy as np
import pandas as pd
import mlflow

from datetime import datetime, timezone
from pathlib import Path
from getpass import getpass
from pyngrok import ngrok

from huggingface_hub import hf_hub_download
from mlflow.tracking import MlflowClient
from mlflow.models.signature import infer_signature

from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from predictive_maintenance.model_selection.model_selection import run_model_selection
from sklearn.exceptions import InconsistentVersionWarning

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# ===============================
# CONFIG
# ===============================
RANDOM_STATE = 42
EXPERIMENT_NAME = "Predictive_Maintenance_Final_Training"
MODEL_REGISTRY_NAME = "PredictiveMaintenanceModel"

MLFLOW_LOCAL_URI = "file:./mlruns"
NGROK_PORT = 5000

HF_DATASET_REPO = "samdurai102024/predictive-maintenance-be"
HF_MODEL_REPO   = "samdurai102024/predictive-maintenance-be"

THRESHOLD = 0.45
DEFAULT_CHAMPION = "XGB"

MODEL_ARTIFACTS = {
    "RF":  "rf_predictive_maintenance_v1.joblib",
    "GBM": "gbm_predictive_maintenance_v1.joblib",
    "XGB": "xgb_predictive_maintenance_v1.joblib"
}

champion_metadata = None

# ===============================
# MLflow + Ngrok
# ===============================
def configure_mlflow():
    if os.getenv("CI"):
        mlflow.set_tracking_uri(MLFLOW_LOCAL_URI)
        mlflow.set_experiment(EXPERIMENT_NAME)
        return

    subprocess.Popen(
        [
            "mlflow", "server",
            "--host", "0.0.0.0",
            "--port", str(NGROK_PORT),
            "--backend-store-uri", MLFLOW_LOCAL_URI,
            "--default-artifact-root", "./mlruns"
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    time.sleep(5)

    ngrok.kill()
    ngrok.set_auth_token(os.getenv("NGROK_TOKEN") or getpass("Enter NGROK token: "))
    tunnel = ngrok.connect(NGROK_PORT)

    mlflow.set_tracking_uri(tunnel.public_url)
    mlflow.set_experiment(EXPERIMENT_NAME)

    print("MLflow UI:", tunnel.public_url)

configure_mlflow()

client = MlflowClient()

# ===============================
# Resolve Champion Algorithm
# ===============================
def get_champion_algo(default=DEFAULT_CHAMPION):
    try:
        mv = client.get_model_version_by_alias(
            name=MODEL_REGISTRY_NAME,
            alias="prod"
        )
        return mv.tags.get("model_family", default)
    except Exception:
        print("Model Registry empty. First-run fallback.")
        return default

champion_algo = get_champion_algo()
print(f"Champion Algorithm: {champion_algo}")

# ===============================
# Load RAW Data (NO preprocessing here)
# ===============================
def load_csv(filename):
    return pd.read_csv(
        hf_hub_download(
            repo_id=HF_DATASET_REPO,
            filename=filename,
            repo_type="dataset"
        )
    )

X_train = load_csv("X_train.csv")
X_val   = load_csv("X_val.csv")
X_test  = load_csv("X_test.csv")

y_train = load_csv("y_train.csv").squeeze().astype(int)
y_val   = load_csv("y_val.csv").squeeze().astype(int)
y_test  = load_csv("y_test.csv").squeeze().astype(int)

print("Data loaded from Hugging Face")

# ===============================
# Load FITTED PIPELINE (preprocessor + model)
# ===============================
def load_tuned_pipeline(algo):
    artifact = MODEL_ARTIFACTS[algo]
    local_path = hf_hub_download(
        repo_id=HF_MODEL_REPO,
        filename=artifact,
        repo_type="model"
    )
    print(f"Loaded fitted pipeline: {artifact}")
    return joblib.load(local_path)

pipeline = load_tuned_pipeline(champion_algo)

# ===============================
# Evaluation Helper (PIPELINE ONLY)
# ===============================
def evaluate(split, X, y):
    probs = pipeline.predict_proba(X)[:, 1]
    preds = (probs >= THRESHOLD).astype(int)

    print(f"\n===== {split.upper()} =====")
    print(confusion_matrix(y, preds))
    print(classification_report(y, preds, digits=4))

    mlflow.log_metric(f"{split}_accuracy", accuracy_score(y, preds))
    mlflow.log_metric(f"{split}_recall", recall_score(y, preds))
    mlflow.log_metric(f"{split}_precision", precision_score(y, preds))
    mlflow.log_metric(f"{split}_f1", f1_score(y, preds))

# ===============================
# MLflow Logging (NO retraining)
# ===============================
with mlflow.start_run(run_name=f"Final_{champion_algo}_Registration"):

    mlflow.set_tags({
        "model_family": champion_algo,
        "stage": "final_registration",
        "threshold": THRESHOLD
    })

    evaluate("train", X_train, y_train)
    evaluate("val", X_val, y_val)
    evaluate("test", X_test, y_test)

    signature = infer_signature(X_train, pipeline.predict(X_train))

    mlflow.sklearn.log_model(
        sk_model=pipeline,
        artifact_path="model",
        signature=signature,
        input_example=X_train.head(5),
        registered_model_name=MODEL_REGISTRY_NAME
    )

# ===============================
# Promote PROD Alias
# ===============================
def write_final_model_info(metadata=None):

    repo_root = Path(os.environ.get("GITHUB_WORKSPACE", os.getcwd()))
    output_path = repo_root / "final_model_info.txt"

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "SUCCESS",
        "champion_model": champion_algo
    }

    if metadata:
        payload["metrics"] = metadata.get("metrics", {})

    output_path.write_text(json.dumps(payload, indent=2))
    print(f"final_model_info.txt written at {output_path}")

    latest = client.get_latest_versions(MODEL_REGISTRY_NAME)[0]

    client.set_registered_model_alias(
        name=MODEL_REGISTRY_NAME,
        alias="prod",
        version=latest.version
    )
try:
    champion_metadata = run_model_selection()
finally:
    write_final_model_info(champion_metadata)

print("Final model registered and PROD alias promoted successfully.")
