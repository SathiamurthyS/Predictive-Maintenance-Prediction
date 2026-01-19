
#funtion to define preprocessor
from prep import build_preprocessor

# for data manipulation
import pandas as pd
import numpy as np
Random_State = 42

from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler
from sklearn.compose import make_column_transformer, ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, RobustScaler

# for model training, tuning, and evaluation
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, recall_score, f1_score, precision_score
from sklearn.metrics import make_scorer

# for creating a folder
import os

# for model serialization
import joblib

#Handling imbalance data with oversampling technique
from sklearn.pipeline import Pipeline

# for creating a folder
import os, getpass

# for hugging face space authentication to upload files
from huggingface_hub import hf_hub_download
from huggingface_hub import login, HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
import mlflow

# Set MLflow tracking URI for PROD (publicly accessible via ngrok)
from pyngrok import ngrok
import mlflow
import time
import subprocess

import os
from pyngrok import ngrok
import mlflow
from getpass import getpass

#Add a simple delay before each MLflow log call
def safe_log_metrics(metric_dict, delay=0.6):
    #Logs multiple metrics to MLflow with a delay to avoid rate limits.
    for name, value in metric_dict.items():
        mlflow.log_metric(name, value)
        time.sleep(delay)

#Add a wrapper around all MLflow calls
def slow_call(fn, *args, delay=0.6, **kwargs):
    result = fn(*args, **kwargs)
    time.sleep(delay)  # delay after EVERY MLflow API call
    return result


EXPERIMENT_NAME = "Predictive_Maintenance_Prod"
MLFLOW_LOCAL_URI = "file:./mlruns"
NGROK_PORT = 5000

#Start MLflow server in background
def start_mlflow_server():
    print("Starting MLflow server on port 5000...")
    subprocess.Popen(
        [
            "mlflow", "server",
            "--host", "0.0.0.0",
            "--port", "5001",
            "--backend-store-uri", "file:./mlruns",
            "--default-artifact-root", "./mlruns"
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(5)  # allow server to boot

def configure_ngrok_and_mlflow():
    is_ci = bool(os.getenv("CI"))

    if is_ci:
        print("Running inside CI — Ngrok disabled")
        mlflow.set_tracking_uri("file:./mlruns")
        mlflow.set_experiment(EXPERIMENT_NAME)
        return None

    # Start MLflow server FIRST
    start_mlflow_server()

    # Ngrok token
    ngrok_token = os.getenv("NGROK_TOKEN")
    if not ngrok_token:
        ngrok_token = getpass("Enter Ngrok Token: ")

    ngrok.set_auth_token(ngrok_token)

    try:
        ngrok.kill()
    except Exception:
        pass

    tunnel = ngrok.connect(5000, proto="http")
    tunnel_url = tunnel.public_url

    print("Ngrok Tunnel:", tunnel_url)
    print(f"MLflow UI: {tunnel_url}")

    mlflow.set_tracking_uri(tunnel_url)
    mlflow.set_experiment(EXPERIMENT_NAME)

    return tunnel_url
    # -------------------------------
    # 2. Get Ngrok token (local only)
    # -------------------------------
    ngrok_token = os.getenv("NGROK_TOKEN")

    if not ngrok_token:
        try:
            print("NGROK_TOKEN not found. Please enter it manually.")
            ngrok_token = getpass("Enter Ngrok Token: ")
        except EOFError:
            raise RuntimeError(
                "NGROK_TOKEN missing and cannot prompt for input. "
                "Set NGROK_TOKEN in environment."
            )

    ngrok.set_auth_token(ngrok_token)

    # -------------------------------
    # 3. Start Ngrok
    # -------------------------------
    print("Running locally — starting Ngrok...")

    # Prevent ERR_NGROK_334
    try:
        ngrok.kill()
    except Exception:
        pass

    tunnel = ngrok.connect(NGROK_PORT, proto="http")
    tunnel_url = tunnel.public_url

    print(f"Ngrok Tunnel Started: {tunnel_url}")
    print(f"Access MLflow UI at: {tunnel_url}/")

    # -------------------------------
    # 4. Configure MLflow
    # -------------------------------
    mlflow.set_tracking_uri(tunnel_url)
    mlflow.set_experiment(EXPERIMENT_NAME)

    return tunnel_url

# ===============================
# Execute configuration
# ===============================
tunnel_url = configure_ngrok_and_mlflow()

print("Final MLflow Tracking URI:", mlflow.get_tracking_uri())
print("Experiment:",
      mlflow.get_experiment_by_name(EXPERIMENT_NAME))

api = HfApi()

# Download the files locally
Xtrain_path = hf_hub_download(
    repo_id="samdurai102024/predictive-maintenance-be",
    filename="X_train.csv",
    repo_type="dataset"    # important!
)
Xval_path = hf_hub_download(
    repo_id="samdurai102024/predictive-maintenance-be",
    filename="X_val.csv",
    repo_type="dataset"    # important!
)
Xtest_path = hf_hub_download(
    repo_id="samdurai102024/predictive-maintenance-be",
    filename="X_test.csv",
    repo_type="dataset"    # important!
)
ytrain_path = hf_hub_download(
    repo_id="samdurai102024/predictive-maintenance-be",
    filename="y_train.csv",
    repo_type="dataset"    # important!
)
yval_path = hf_hub_download(
    repo_id="samdurai102024/predictive-maintenance-be",
    filename="y_val.csv",
    repo_type="dataset"    # important!
)
ytest_path = hf_hub_download(
    repo_id="samdurai102024/predictive-maintenance-be",
    filename="y_test.csv",
    repo_type="dataset"    # important!
)

X_train = pd.read_csv(Xtrain_path)
X_val = pd.read_csv(Xval_path)
X_test = pd.read_csv(Xtest_path)
y_train = pd.read_csv(ytrain_path)
y_val = pd.read_csv(yval_path)
y_test = pd.read_csv(ytest_path)

# Convert target values (not column names) to integer
y_train = y_train.astype(int)
y_val = y_val.astype(int)
y_test = y_test.astype(int)

# Set the class weight to handle class imbalance
class_weight = y_train.value_counts()[0] / y_train.value_counts()[1]
#class_weight

# Define categorical and numerical column names (these are the original, raw column names)
categorical_cols = X_train.select_dtypes(include=['category', 'object']).columns
numerical_cols = X_train.select_dtypes(include=[np.number]).columns

# Convert target values (not column names) to integer
# Ensure y is DataFrame
def ensure_target_dataframe(y, name="target"):
    """
    Ensure y is a pandas DataFrame with integer values.
    """
    # Convert to DataFrame if not already
    if not isinstance(y, pd.DataFrame):
        y = pd.DataFrame(y, columns=[name])

    # Convert values to int
    y = y.astype(int)

    return y

# Apply to train/val/test
y_train = ensure_target_dataframe(y_train, name="Engine_Condition")
y_val   = ensure_target_dataframe(y_val, name="Engine_Condition")
y_test  = ensure_target_dataframe(y_test, name="Engine_Condition")

# Set the clas weight to handle class imbalance
class_weight = y_train.value_counts()[0] / y_train.value_counts()[1]
class_weight

# Define base XGBoost model
xgb_model = xgb.XGBClassifier(
    random_state=Random_State,
    scale_pos_weight = class_weight,
    eval_metric="logloss"
)

# Define categorical and numerical column names (these are the original, raw column names)
categorical_cols = X_train.select_dtypes(include=['category', 'object']).columns
numerical_cols = X_train.select_dtypes(include=[np.number]).columns

# Build preprocessor (NOT FITTED yet)
preprocessor = build_preprocessor(
    numerical_cols = numerical_cols
)

# Create pipeline using pipeline
xgb_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier',xgb_model)
])

# Define hyperparameter grid
param_grid ={
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
# Type of scoring used to compare parameter combinations
scorer = make_scorer(recall_score, average='macro')

with mlflow.start_run():
    # Hyperparameter tuning
    grid_search = GridSearchCV(xgb_pipeline, param_grid, scoring = scorer, cv = 5, n_jobs = -1)
    grid_search.fit(X_train, y_train)

    # Log all parameter combinations and their mean test scores
    results = grid_search.cv_results_
    for i in range(len(results['params'])):
        param_set = results['params'][i]
        mean_score = results['mean_test_score'][i]
        std_score = results['std_test_score'][i]

        # Log each combination as a separate MLflow run
        with slow_call(mlflow.start_run, nested=True, delay=0.7):
          slow_call(mlflow.log_params, param_set, delay=0.5)
          slow_call(mlflow.log_metric, "mean_test_score", mean_score, delay=0.5)
          slow_call(mlflow.log_metric, "std_test_score", std_score, delay=0.5)

        #with mlflow.start_run(nested=True):
         #   mlflow.log_params(param_set)
         #   mlflow.log_metric("mean_test_score", mean_score)
         #   mlflow.log_metric("std_test_score", std_score)


    # Log best parameters separately in main run
    mlflow.log_params(grid_search.best_params_)

    # Store and evaluate the best model
    best_model = grid_search.best_estimator_

    classification_threshold = 0.45

    y_pred_train_proba = best_model.predict_proba(X_train)[:, 1]
    y_pred_train = (y_pred_train_proba >= classification_threshold).astype(int)

    y_pred_val_proba = best_model.predict_proba(X_val)[:, 1]
    y_pred_val = (y_pred_val_proba >= classification_threshold).astype(int)

    y_pred_test_proba = best_model.predict_proba(X_test)[:, 1]
    y_pred_test = (y_pred_test_proba >= classification_threshold).astype(int)

    train_report = classification_report(y_train, y_pred_train, output_dict=True)
    val_report = classification_report(y_val, y_pred_val, output_dict=True)
    test_report = classification_report(y_test, y_pred_test, output_dict=True)

    safe_log_metrics({
    "train_accuracy": train_report['accuracy'],
    "train_precision": train_report['1']['precision'],
    "train_recall": train_report['1']['recall'],
    "train_f1_score": train_report['1']['f1-score'],

    "val_accuracy": val_report['accuracy'],
    "val_precision": val_report['1']['precision'],
    "val_recall": val_report['1']['recall'],
    "val_f1_score": val_report['1']['f1-score'],

    "test_accuracy": test_report['accuracy'],
    "test_precision": test_report['1']['precision'],
    "test_recall": test_report['1']['recall'],
    "test_f1_score": test_report['1']['f1-score']
    })

    # Save the model locally
    model_path = "best_predictive_maintenance_prediction_v1.joblib"
    joblib.dump(best_model, model_path)

    # Log the model artifact
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"Model saved as artifact at: {model_path}")

    # Upload to Hugging Face
    repo_id = "samdurai102024/predictive-maintenance-be"
    repo_type = "model"

    # Step 1: Check if the space exists
    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type)
        print(f"Space '{repo_id}' already exists. Using it.")
    except RepositoryNotFoundError:
        print(f"Space '{repo_id}' not found. Creating new space...")
        create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
        print(f"Space '{repo_id}' created.")

# create_repo("churn-model", repo_type="model", private=False)
    api.upload_file(
        path_or_fileobj = "best_predictive_maintenance_prediction_v1.joblib",
        path_in_repo = "best_predictive_maintenance_prediction_v1.joblib",
        repo_id=repo_id,
        repo_type=repo_type,
    )
#Commit changes
    api.create_commit(
        repo_id=repo_id,
        repo_type=repo_type,
        operations=[],
        commit_message="Force empty commit",
        )
