
# ============================================================
# Retraining Controller (Self-Contained, Automation-Safe)
# ============================================================

import os
import sys
import subprocess
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timezone
import json

from huggingface_hub import hf_hub_download
from sklearn.metrics import recall_score

from pathlib import Path

# ============================================================
# Path Resolution (MUST be defined before use)
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent  # predictive_maintenance/

# ============================================================
# CONFIG
# ============================================================

HF_DATASET_REPO = "samdurai102024/predictive-maintenance-be"
HF_MODEL_REPO   = "samdurai102024/predictive-maintenance-be"

PRODUCTION_MODEL_PATH = "production/model.joblib"

THRESHOLD = 0.45
MIN_ACCEPTABLE_RECALL = 0.96

# Training commands (RELATIVE TO THIS FILE)
TRAIN_RF_CMD = [
    "python",
    str(BASE_DIR.parent / "model_building" / "train_rf.py")
]

TRAIN_GBM_CMD = [
    "python",
    str(BASE_DIR.parent / "model_building" / "train_gbm.py")
]

TRAIN_XGB_CMD = [
    "python",
    str(BASE_DIR.parent / "model_building" / "train_xgb.py")
]

MODEL_SELECT_CMD = [
    "python",
    str(BASE_DIR / "model_selection.py")
]

# ============================================================
# Utilities
# ============================================================

def load_csv(filename):
    return pd.read_csv(
        hf_hub_download(
            repo_id=HF_DATASET_REPO,
            filename=filename,
            repo_type="dataset"
        )
    )

def load_production_model():
    local_path = hf_hub_download(
        repo_id=HF_MODEL_REPO,
        filename=PRODUCTION_MODEL_PATH,
        repo_type="model"
    )
    return joblib.load(local_path)

# ============================================================
# Compute Validation Recall from CURRENT Production Model
# ============================================================

def compute_val_recall():
    model = load_production_model()

    X_val = load_csv("X_val.csv")
    y_val = load_csv("y_val.csv").squeeze().astype(int)

    try:
        probs = model.predict_proba(X_val)[:, 1]
    except AttributeError:
        probs = model.predict(X_val)

    preds = (probs >= THRESHOLD).astype(int)
    return recall_score(y_val, preds)

# ============================================================
# Retraining Decision
# ============================================================

def run_full_retraining():
    print("\n Retraining triggered")
    print("Reason           : performance_drop")
    print(f"Timestamp        : {datetime.now(timezone.utc).isoformat()}")
    print("Scope            : RF + GBM + XGB + Model Selection\n")

    try:

        env = os.environ.copy()
        env["AUTO_RETRAIN"] = "1"

        print("▶ Training Random Forest")
        subprocess.run(TRAIN_RF_CMD, check=True, env=env)

        print("▶ Training Gradient Boosting")
        subprocess.run(TRAIN_GBM_CMD, check=True, env=env)

        print("▶ Training XGBoost")
        subprocess.run(TRAIN_XGB_CMD, check=True, env=env)

        print("▶ Selecting Best Model")
        subprocess.run(MODEL_SELECT_CMD, check=True)

        print("\n Retraining completed successfully")

    except subprocess.CalledProcessError as e:
        print("\n Retraining failed")
        print(e)
        raise RuntimeError("Retraining failed")

# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    print("\n==============================")
    print("RETRAINING CONTROLLER STARTED")
    print("==============================")

    val_recall = compute_val_recall()

    print(f"Validation Recall : {val_recall:.4f}")
    print(f"Threshold         : {MIN_ACCEPTABLE_RECALL:.2f}")

    if val_recall < MIN_ACCEPTABLE_RECALL:
        run_full_retraining()
    else:
        print("\n No retraining required — model is healthy")
retraining_summary = {
    "pipeline": "Predictive Maintenance",
    "trigger_reason": "performance_drop",
    "validation_recall": float(val_recall),
    "threshold": float(THRESHOLD),
    "retrained_models": ["RandomForest", "GradientBoosting", "XGBoost"],
    "best_model": best_model_name,
    "timestamp_utc": datetime.utcnow().isoformat()
}

with open("retraining_summary.json", "w") as f:
    json.dump(retraining_summary, f, indent=2)

print("Retraining summary artifact generated")
