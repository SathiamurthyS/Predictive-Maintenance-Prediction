
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
from huggingface_hub import hf_hub_download, HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

import xgboost as xgb

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    recall_score,
    precision_score,
    f1_score,
    accuracy_score,
    make_scorer
)

from prep import build_preprocessor

# ===============================
# Constants
# ===============================
RANDOM_STATE = 42
EXPERIMENT_NAME = "Predictive_Maintenance_XGB_GridSearch"
MLFLOW_LOCAL_URI = "file:./mlruns"
NGROK_PORT = 5000

# ===============================
# MLflow + Ngrok Setup
# ===============================
def start_mlflow_server():
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

def configure_mlflow():
    is_ci = bool(os.getenv("CI"))

    if is_ci:
        mlflow.set_tracking_uri(MLFLOW_LOCAL_URI)
        mlflow.set_experiment(EXPERIMENT_NAME)
        return

    start_mlflow_server()

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
# Load Data
# ===============================
def load_csv(repo_id, filename):
    return pd.read_csv(
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset"
        )
    )

REPO_ID = "samdurai102024/predictive-maintenance-be"

X_train = load_csv(REPO_ID, "X_train.csv")
X_val   = load_csv(REPO_ID, "X_val.csv")
X_test  = load_csv(REPO_ID, "X_test.csv")

y_train = load_csv(REPO_ID, "y_train.csv")#.squeeze().astype(int)
y_val   = load_csv(REPO_ID, "y_val.csv")#.squeeze().astype(int)
y_test  = load_csv(REPO_ID, "y_test.csv")#.squeeze().astype(int)

# ===============================
# Preprocessor
# ===============================
numerical_cols = X_train.select_dtypes(include=[np.number]).columns

preprocessor = build_preprocessor(
    numerical_cols=numerical_cols
)

# ===============================
# Class Imbalance Handling
# ===============================
scale_pos_weight = (
    y_train.value_counts()[0] / y_train.value_counts()[1]
)

# ===============================
# Model + Pipeline
# ===============================
xgb_model = xgb.XGBClassifier(
    random_state=RANDOM_STATE,
    scale_pos_weight=scale_pos_weight,
    eval_metric="logloss",
    n_jobs=-1,
    use_label_encoder=False
)

xgb_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", xgb_model)
])

# ===============================
# Grid Search
# ===============================
param_grid = {
    "classifier__n_estimators": [700, 800],
    "classifier__learning_rate": [0.07, 0.08],
    "classifier__max_depth": [4, 5],
    "classifier__min_child_weight": [7, 8],
    "classifier__gamma": [0.4, 0.5],
    "classifier__subsample": [0.4, 0.5],
    "classifier__colsample_bytree": [0.4, 0.5],
    "classifier__reg_alpha": [0.2, 0.3],
    "classifier__reg_lambda": [4, 5],
    "classifier__scale_pos_weight": [2],
}

scorer = make_scorer(recall_score, average='macro')

# ===============================
# MLflow Training
# ===============================
with mlflow.start_run(run_name="XGB_GridSearch"):

    mlflow.set_tags({
        "model_family": "XGBoost",
        "tuning_method": "gridsearch",
        "sampling": "none",
        "primary_metric": "recall"
    })

    grid = GridSearchCV(
        xgb_pipeline,
        param_grid=param_grid,
        scoring=scorer,
        cv=5,
        n_jobs=-1,
        verbose=1
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

    # Save model
    model_path = "xgb_predictive_maintenance_v1.joblib"
    joblib.dump(best_model, model_path)
    mlflow.log_artifact(model_path, artifact_path="model")

# ===============================
# Upload to Hugging Face
# ===============================
api = HfApi()
MODEL_REPO = "samdurai102024/predictive-maintenance-be"

try:
    api.repo_info(repo_id=MODEL_REPO, repo_type="model")
except RepositoryNotFoundError:
    create_repo(repo_id=MODEL_REPO, repo_type="model", private=False)

api.upload_file(
    path_or_fileobj=model_path,
    path_in_repo=model_path,
    repo_id=MODEL_REPO,
    repo_type="model"
)
