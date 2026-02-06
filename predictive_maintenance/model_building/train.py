
# ===============================
# Imports
# ===============================
import os
import joblib
import numpy as np
import pandas as pd
import mlflow
from datetime import datetime
from pathlib import Path
from datetime import timezone
import json
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

from prep import build_preprocessor
from mlflow.models.signature import infer_signature

# ===============================
# CONFIG
# ===============================
RANDOM_STATE = 42
EXPERIMENT_NAME = "Predictive_Maintenance_Final_Training"
MODEL_REGISTRY_NAME = "PredictiveMaintenanceModel"

MLFLOW_LOCAL_URI = "file:./mlruns"

HF_DATASET_REPO = "samdurai102024/predictive-maintenance-be"
HF_MODEL_REPO   = "samdurai102024/predictive-maintenance-be"

THRESHOLD = 0.45
DEFAULT_CHAMPION = "XGB"

MODEL_ARTIFACTS = {
    "RF":  "rf_predictive_maintenance_v1.joblib",
    "GBM": "gbm_predictive_maintenance_v1.joblib",
    "XGB": "xgb_predictive_maintenance_v1.joblib"
}

PRODUCTION_MODEL_PATH = "production/model.joblib"
ARCHIVE_PATH = "archive"

# ===============================
# MLflow Configuration (LOCAL ONLY)
# ===============================
mlflow.set_tracking_uri(MLFLOW_LOCAL_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

# ===============================
# Resolve Champion Algorithm
# ===============================
def get_champion_algo(default=DEFAULT_CHAMPION):
    """
    Use MLflow registry ONLY as metadata.
    Falls back safely if registry is empty.
    """
    try:
        client = MlflowClient()
        versions = client.get_latest_versions(
            name=MODEL_REGISTRY_NAME,
            stages=["Production"]
        )
        if not versions:
            print("Model Registry empty. First-run fallback.")
            return default

        return versions[0].tags.get("model_family", default)

    except Exception:
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
# Load Tuned Model from HF
# ===============================
def load_tuned_pipeline(algo):
    artifact_name = MODEL_ARTIFACTS[algo]

    local_path = hf_hub_download(
        repo_id=HF_MODEL_REPO,
        filename=artifact_name,
        repo_type="model"
    )

    print(f"Loaded tuned model artifact: {artifact_name}")
    return joblib.load(local_path)


tuned_pipeline = load_tuned_pipeline(champion_algo)

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
# Training + Evaluation
# ===============================
with mlflow.start_run(run_name=f"Final_{champion_algo}_Training"):

    mlflow.set_tags({
        "model_family": champion_algo,
        "stage": "final_training",
        "threshold": THRESHOLD
    })

    pipeline.fit(X_train, y_train)

    def evaluate(split, X, y):
        try:
            probs = pipeline.predict_proba(X)[:, 1]
        except AttributeError:
            probs = pipeline.predict(X)

        preds = (probs >= THRESHOLD).astype(int)

        print(f"\n===== {split.upper()} =====")
        print(confusion_matrix(y, preds))
        print(classification_report(y, preds, digits=4))

        mlflow.log_metric(f"{split}_accuracy", accuracy_score(y, preds))
        mlflow.log_metric(f"{split}_recall", recall_score(y, preds))
        mlflow.log_metric(f"{split}_precision", precision_score(y, preds))
        mlflow.log_metric(f"{split}_f1", f1_score(y, preds))

    evaluate("train", X_train, y_train)
    evaluate("val", X_val, y_val)
    evaluate("test", X_test, y_test)

    signature = infer_signature(X_train, pipeline.predict(X_train))

    mlflow.sklearn.log_model(
        pipeline,
        artifact_path="model",
        signature=signature,
        registered_model_name=MODEL_REGISTRY_NAME
    )

# ===============================
# Push FINAL Model to Hugging Face (Automation Safe)
# ===============================
api = HfApi()
timestamp = datetime.now().strftime("%Y_%m_%d_%H%M")

# ===============================
# Push FINAL Model to Hugging Face
# ===============================
os.makedirs(ARCHIVE_PATH, exist_ok=True)

# Archive champion
archive_file = f"{ARCHIVE_PATH}/{champion_algo.lower()}_{timestamp}.joblib"
joblib.dump(pipeline, archive_file)

api.upload_file(
    path_or_fileobj=archive_file,
    path_in_repo=archive_file,
    repo_id=HF_MODEL_REPO,
    repo_type="model",
    commit_message=f"Archive {champion_algo} model ({timestamp})"
)

# Promote to production (stable alias)
joblib.dump(pipeline, "model.joblib")

api.upload_file(
    path_or_fileobj="model.joblib",
    path_in_repo=PRODUCTION_MODEL_PATH,
    repo_id=HF_MODEL_REPO,
    repo_type="model",
    commit_message=f"Promote {champion_algo} to production"
)

def write_final_model_info(champion_metadata: dict | None = None):
    """
    Always writes final_model_info.txt for CI artifact upload.
    Safe even if champion metadata is missing.
    """

    output_path = Path("final_model_info.txt")

    base_payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "SUCCESS",
    }

    if champion_metadata:
        base_payload.update({
            "champion_model": champion_metadata.get("model_name", "UNKNOWN"),
            "run_id": champion_metadata.get("run_id", "UNKNOWN"),
            "val_recall": champion_metadata.get("metrics", {}).get("val_recall"),
            "val_precision": champion_metadata.get("metrics", {}).get("val_precision"),
            "val_f1": champion_metadata.get("metrics", {}).get("val_f1"),
            "val_accuracy": champion_metadata.get("metrics", {}).get("val_accuracy"),
        })
    else:
        base_payload["note"] = "Champion metadata unavailable; fallback file created"

    output_path.write_text(
        json.dumps(base_payload, indent=2),
        encoding="utf-8"
    )

    print(f"final_model_info.txt written at {output_path.resolve()}")

print("Final production model promoted to Hugging Face successfully.")
