
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

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    recall_score,
    precision_score,
    f1_score,
    accuracy_score,
    classification_report,
    make_scorer
)

from prep import build_preprocessor

# ===============================
# Constants
# ===============================
RANDOM_STATE = 42
EXPERIMENT_NAME = "Predictive_Maintenance_RF_GridSearch"
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
        hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")
    )

REPO_ID = "samdurai102024/predictive-maintenance-be"

X_train = load_csv(REPO_ID, "X_train.csv")
X_val   = load_csv(REPO_ID, "X_val.csv")
X_test  = load_csv(REPO_ID, "X_test.csv")

y_train = load_csv(REPO_ID, "y_train.csv").squeeze().astype(int)
y_val   = load_csv(REPO_ID, "y_val.csv").squeeze().astype(int)
y_test  = load_csv(REPO_ID, "y_test.csv").squeeze().astype(int)

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
rf = RandomForestClassifier(
    random_state=RANDOM_STATE,
    n_jobs=-1
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", rf)
])

# ===============================
# Grid Search
# ===============================
param_grid = {
    "classifier__n_estimators": [450, 470],
    "classifier__max_depth": [5, 6],
    "classifier__min_samples_split": [7, 8],
    "classifier__min_samples_leaf": [5, 6],
    "classifier__max_features": [0.3, 0.4],
    "classifier__class_weight": [{0: 0.63, 1: 0.37}],
}

scorer = make_scorer(recall_score, pos_label=1)

# ===============================
# MLflow Training
# ===============================
with mlflow.start_run(run_name="RF_GridSearch"):

    mlflow.set_tags({
        "model_family": "RandomForest",
        "tuning_method": "gridsearch",
        "sampling": "none",
        "primary_metric": "recall"
    })

    grid = GridSearchCV(
        pipeline,
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
    model_path = "rf_predictive_maintenance_v1.joblib"
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
