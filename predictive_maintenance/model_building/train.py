
# ===============================
# Imports
# ===============================
import os
import time
import subprocess
import joblib
import numpy as np
import pandas as pd
import mlflow

from getpass import getpass
from pyngrok import ngrok
from mlflow.tracking import MlflowClient
from huggingface_hub import hf_hub_download, HfApi

from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from mlflow.models.signature import infer_signature
from prep import build_preprocessor

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

ARCHIVE_PATH = "archive"
os.makedirs(ARCHIVE_PATH, exist_ok=True)

MODEL_ARTIFACTS = {
    "RF":  "rf_predictive_maintenance_v1.joblib",
    "GBM": "gbm_predictive_maintenance_v1.joblib",
    "XGB": "xgb_predictive_maintenance_v1.joblib"
}

# ===============================
# MLflow + Ngrok
# ===============================
def configure_mlflow():
    if os.getenv("CI"):
        mlflow.set_tracking_uri(MLFLOW_LOCAL_URI)
        mlflow.set_experiment(EXPERIMENT_NAME)
        return None

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

    print("MLflow Tracking UI:", tunnel.public_url)
    return tunnel.public_url


MLFLOW_UI_URL = configure_mlflow()

# ===============================
# Resolve Champion Algorithm
# ===============================
def get_champion_algo(default=DEFAULT_CHAMPION):
    try:
        client = MlflowClient()
        versions = client.search_model_versions(
            f"name='{MODEL_REGISTRY_NAME}'"
        )
        if not versions:
            print("Model Registry empty. First-run fallback.")
            return default

        return versions[-1].tags.get("model_family", default)

    except Exception as e:
        print(f"Registry unavailable ({e}). Falling back.")
        return default


champion_algo = get_champion_algo()
print(f"Champion Algorithm Selected: {champion_algo}")

# ===============================
# Load Data (HF = Source of Truth)
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
# Load Tuned Model
# ===============================
artifact_name = MODEL_ARTIFACTS[champion_algo]
model_path = hf_hub_download(
    repo_id=HF_MODEL_REPO,
    filename=artifact_name,
    repo_type="model"
)

tuned_pipeline = joblib.load(model_path)
print(f"Loaded tuned model artifact: {artifact_name}")

# ===============================
# Build Final Pipeline
# ===============================
numerical_cols = X_train.select_dtypes(include=[np.number]).columns
preprocessor = build_preprocessor(numerical_cols=numerical_cols)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", tuned_pipeline.named_steps["classifier"])
])

# ===============================
# Training + MLflow Tracking
# ===============================
with mlflow.start_run(run_name=f"Final_{champion_algo}_Training") as run:

    mlflow.set_tags({
        "model_family": champion_algo,
        "deployment_status": "production",
        "production_source": "huggingface",
        "threshold": THRESHOLD
    })

    pipeline.fit(X_train, y_train)

    def evaluate(split, X, y):
        probs = pipeline.predict_proba(X)[:, 1]
        preds = (probs >= THRESHOLD).astype(int)

        mlflow.log_metric(f"{split}_accuracy", accuracy_score(y, preds))
        mlflow.log_metric(f"{split}_recall", recall_score(y, preds))
        mlflow.log_metric(f"{split}_precision", precision_score(y, preds))
        mlflow.log_metric(f"{split}_f1", f1_score(y, preds))

        print(f"\n===== {split.upper()} =====")
        print(confusion_matrix(y, preds))
        print(classification_report(y, preds, digits=4))

    evaluate("train", X_train, y_train)
    evaluate("val", X_val, y_val)
    evaluate("test", X_test, y_test)

    signature = infer_signature(X_train, pipeline.predict(X_train))

    mlflow.sklearn.log_model(
        pipeline,
        name="model",
        signature=signature,
        input_example=X_train.head(5),
        registered_model_name=MODEL_REGISTRY_NAME
    )

    run_id = run.info.run_id

# ===============================
# Output for Stakeholders
# ===============================
print("\n==============================")
print("TRAINING & TRACKING COMPLETE")
print("==============================")

if MLFLOW_UI_URL:
    print(f"MLflow Experiment UI : {MLFLOW_UI_URL}")
    print(f"MLflow Run URL      : {MLFLOW_UI_URL}/#/experiments/0/runs/{run_id}")

print(f"Registered Model    : {MODEL_REGISTRY_NAME}")
print(f"Champion Algorithm  : {champion_algo}")
print("Production Model    : Served from Hugging Face")
