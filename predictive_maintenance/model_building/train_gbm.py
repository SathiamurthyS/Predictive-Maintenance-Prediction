
# ===============================
# Path setup (single, correct)
# ===============================
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# ===============================
# Imports
# ===============================
import time
import subprocess
import joblib
import numpy as np
import pandas as pd
import mlflow

from getpass import getpass
from pyngrok import ngrok
from huggingface_hub import hf_hub_download, HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    recall_score,
    precision_score,
    f1_score,
    accuracy_score,
    make_scorer
)

from predictive_maintenance.model_building.prep import build_preprocessor

# ===============================
# Constants
# ===============================
RANDOM_STATE = 42
EXPERIMENT_NAME = "Predictive_Maintenance_GBM_GridSearch"
MLFLOW_LOCAL_URI = "file:./mlruns"
NGROK_PORT = 5000
REPO_ID = "samdurai102024/predictive-maintenance-be"

# ===============================
# MLflow + Ngrok configuration
# ===============================
def configure_mlflow():
    is_ci = bool(os.getenv("CI")) or bool(os.getenv("AUTO_RETRAIN"))

    if is_ci:
        mlflow.set_tracking_uri(MLFLOW_LOCAL_URI)
        mlflow.set_experiment(EXPERIMENT_NAME)
        print("Running in non-interactive mode (CI / Auto Retrain)")
        return

    # Start MLflow server
    subprocess.Popen(
        [
            "mlflow", "server",
            "--host", "0.0.0.0",
            "--port", str(NGROK_PORT),
            "--backend-store-uri", MLFLOW_LOCAL_URI,
            "--default-artifact-root", "./mlruns",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(5)

    # Ask for NGROK token interactively if not present
    ngrok_token = os.getenv("NGROK_TOKEN") or getpass("Enter NGROK token: ")
    ngrok.set_auth_token(ngrok_token)

    try:
        ngrok.kill()
    except Exception:
        pass

    tunnel = ngrok.connect(NGROK_PORT, proto="http")
    mlflow.set_tracking_uri(tunnel.public_url)
    mlflow.set_experiment(EXPERIMENT_NAME)

    print(f"MLflow UI: {tunnel.public_url}")

configure_mlflow()

# ===============================
# Load data
# ===============================
def load_csv(filename):
    return pd.read_csv(
        hf_hub_download(
            repo_id=REPO_ID,
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

# ===============================
# Preprocessor
# ===============================
numerical_cols = X_train.select_dtypes(include=[np.number]).columns

preprocessor = build_preprocessor(
    numerical_cols=numerical_cols
)

# ===============================
# Model + Pipeline
# ===============================
gbm_model = GradientBoostingClassifier(
    random_state=RANDOM_STATE
)

gbm_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", gbm_model),
    ]
)

# ===============================
# Grid Search
# ===============================
param_grid = {
    "classifier__n_estimators": [100, 150],
    "classifier__learning_rate": [0.05, 0.06],
    "classifier__max_depth": [3, 4],
    "classifier__min_samples_split": [8, 9],
    "classifier__min_samples_leaf": [4, 6],
    "classifier__subsample": [0.8, 1.0],
    "classifier__max_features": [0.8, 1.0],
}

scorer = make_scorer(recall_score, average="binary")

# ===============================
# Training + MLflow logging
# ===============================
with mlflow.start_run(run_name="GBM_GridSearch"):

    mlflow.set_tags({
        "model_family": "GradientBoosting",
        "tuning_method": "gridsearch",
        "primary_metric": "recall",
    })

    grid = GridSearchCV(
        estimator=gbm_pipeline,
        param_grid=param_grid,
        scoring=scorer,
        cv=5,
        n_jobs=-1,
        verbose=1,
    )

    grid.fit(X_train, y_train)

    mlflow.log_params(grid.best_params_)
    best_model = grid.best_estimator_

    def log_metrics(split, X, y):
        y_pred = best_model.predict(X)
        mlflow.log_metric(f"{split}_accuracy", accuracy_score(y, y_pred))
        mlflow.log_metric(f"{split}_recall", recall_score(y, y_pred))
        mlflow.log_metric(f"{split}_precision", precision_score(y, y_pred))
        mlflow.log_metric(f"{split}_f1", f1_score(y, y_pred))

    log_metrics("train", X_train, y_train)
    log_metrics("val", X_val, y_val)
    log_metrics("test", X_test, y_test)

    model_path = "gbm_predictive_maintenance_v1.joblib"
    joblib.dump(best_model, model_path)
    mlflow.log_artifact(model_path, artifact_path="model")

# ===============================
# Upload model to Hugging Face
# ===============================
api = HfApi()

try:
    api.repo_info(repo_id=REPO_ID, repo_type="model")
except RepositoryNotFoundError:
    create_repo(repo_id=REPO_ID, repo_type="model", private=False)

api.upload_file(
    path_or_fileobj=model_path,
    path_in_repo=model_path,
    repo_id=REPO_ID,
    repo_type="model",
    commit_message="GBM retraining artifact",
)

print("GBM training completed successfully.")
