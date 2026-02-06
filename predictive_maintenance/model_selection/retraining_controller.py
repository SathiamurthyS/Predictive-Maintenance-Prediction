
# ============================================================
# Retraining Controller (Automation-Safe, CI-Compatible)
# ============================================================

import os
import subprocess
import joblib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download
from sklearn.metrics import recall_score

# ============================================================
# Path Resolution
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent  # predictive_maintenance/

# ============================================================
# CONFIG
# ============================================================

HF_DATASET_REPO = "samdurai102024/predictive-maintenance-be"
HF_MODEL_REPO   = "samdurai102024/predictive-maintenance-be"

PRODUCTION_MODEL_PATH = "production/model.joblib"
CHAMPION_MODEL_PATH   = "production/champion.model"

THRESHOLD = 0.45
MIN_ACCEPTABLE_RECALL = 0.96

# ============================================================
# Training Commands
# ============================================================

TRAIN_RF_CMD = ["python", str(PROJECT_ROOT / "model_building" / "train_rf.py")]
TRAIN_GBM_CMD = ["python", str(PROJECT_ROOT / "model_building" / "train_gbm.py")]
TRAIN_XGB_CMD = ["python", str(PROJECT_ROOT / "model_building" / "train_xgb.py")]
MODEL_SELECT_CMD = ["python", str(BASE_DIR / "model_selection.py")]

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
    path = hf_hub_download(
        repo_id=HF_MODEL_REPO,
        filename=PRODUCTION_MODEL_PATH,
        repo_type="model"
    )
    return joblib.load(path)

def load_champion_model():
    path = hf_hub_download(
        repo_id=HF_MODEL_REPO,
        filename=CHAMPION_MODEL_PATH,
        repo_type="model"
    )
    return joblib.load(path)

# ============================================================
# Compute Validation Recall (Current Production Model)
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
# Retraining Pipeline
# ============================================================

def run_full_retraining():
    print("\n Retraining triggered")
    print(f"Timestamp : {datetime.now(timezone.utc).isoformat()}")
    print("Scope     : RF + GBM + XGB + Model Selection\n")

    env = os.environ.copy()
    env["AUTO_RETRAIN"] = "1"

    subprocess.run(TRAIN_RF_CMD, check=True, env=env)
    subprocess.run(TRAIN_GBM_CMD, check=True, env=env)
    subprocess.run(TRAIN_XGB_CMD, check=True, env=env)
    subprocess.run(MODEL_SELECT_CMD, check=True)

    print("\n Retraining completed successfully")

# ============================================================
# Main
# ============================================================

def main():
    print("\n==============================")
    print("RETRAINING CONTROLLER STARTED")
    print("==============================")

    val_recall = compute_val_recall()

    print(f"Validation Recall : {val_recall:.4f}")
    print(f"Minimum Required  : {MIN_ACCEPTABLE_RECALL:.2f}")

    retrained = False

    if val_recall < MIN_ACCEPTABLE_RECALL:
        run_full_retraining()
        retrained = True
    else:
        print("\n No retraining required — model is healthy")

    # ========================================================
    # Generate Summary ONLY if retraining occurred
    # ========================================================

    if retrained:
        champion = load_champion_model()

        best_model_name = champion.get("model_name")
        if best_model_name is None:
            raise RuntimeError("Champion model missing 'model_name'")

        summary = {
            "pipeline": "Predictive Maintenance",
            "trigger_reason": "performance_drop",
            "validation_recall": float(val_recall),
            "decision_threshold": float(THRESHOLD),
            "retrained_models": ["RandomForest", "GradientBoosting", "XGBoost"],
            "best_model": best_model_name,
            "timestamp_utc": datetime.utcnow().isoformat()
        }

        with open("retraining_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        print(" Retraining summary artifact generated")

if __name__ == "__main__":
    main()
