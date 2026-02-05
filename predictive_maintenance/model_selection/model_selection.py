
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from datetime import datetime
from mlflow.tracking import MlflowClient
from sklearn.metrics import recall_score, precision_score, f1_score

# ===============================
# CONFIG
# ===============================
EXPERIMENTS = {
    "RF": "Predictive_Maintenance_RF_GridSearch",
    "GBM": "Predictive_Maintenance_GBM_GridSearch",
    "XGB": "Predictive_Maintenance_XGB_GridSearch"
}

PRIMARY_METRIC = "val_recall"
MODEL_NAME = "PredictiveMaintenanceModel"
MIN_RECALL = 0.85
TRACKING_URI = "file:./mlruns"

mlflow.set_tracking_uri(TRACKING_URI)
client = MlflowClient()

# ===============================
# STEP 1: CROSS-EXPERIMENT COMPARISON
# ===============================
def fetch_best_runs():
    records = []

    for algo, exp_name in EXPERIMENTS.items():
        exp = mlflow.get_experiment_by_name(exp_name)
        if exp is None:
            continue

        runs = mlflow.search_runs(
            experiment_ids=[exp.experiment_id],
            order_by=[f"metrics.{PRIMARY_METRIC} DESC"],
            max_results=1
        )

        if runs.empty:
            continue

        row = runs.iloc[0]

        records.append({
            "model": algo,
            "run_id": row.run_id,
            "model_uri": f"runs:/{row.run_id}/model",
            "val_recall": row.get("metrics.val_recall", 0.0),
            "val_precision": row.get("metrics.val_precision", 0.0),
            "val_f1": row.get("metrics.val_f1", 0.0),
            "val_accuracy": row.get("metrics.val_accuracy", 0.0),
        })

    if not records:
        raise RuntimeError("No candidate runs found across experiments")

    return pd.DataFrame(records).sort_values(
        by=PRIMARY_METRIC, ascending=False
    )

# ===============================
# STEP 2: MODEL REGISTRY PROMOTION
# ===============================
def register_and_promote(df):
    champion = df.iloc[0]
    challenger = df.iloc[1] if len(df) > 1 else None

    def _register(row, stage):
        mv = mlflow.register_model(
            model_uri=row.model_uri,
            name=MODEL_NAME
        )

        client.set_model_version_tag(
            name=MODEL_NAME,
            version=mv.version,
            key="model_family",
            value=row.model
        )

        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=mv.version,
            stage=stage,
            archive_existing_versions=True
        )

        return mv.version

    champion_version = _register(champion, "Production")

    challenger_version = None
    if challenger is not None:
        challenger_version = _register(challenger, "Staging")

    return champion, challenger, champion_version, challenger_version

# ===============================
# STEP 3: THRESHOLD OPTIMIZATION
# ===============================
def tune_threshold(model, X_val, y_val):
    probs = model.predict_proba(X_val)[:, 1]
    thresholds = np.arange(0.1, 0.9, 0.05)

    rows = []
    for t in thresholds:
        preds = (probs >= t).astype(int)
        rows.append({
            "threshold": t,
            "recall": recall_score(y_val, preds),
            "precision": precision_score(y_val, preds),
            "f1": f1_score(y_val, preds)
        })

    df = pd.DataFrame(rows)

    eligible = df[df.recall >= MIN_RECALL]
    if eligible.empty:
        raise RuntimeError(
            f"No threshold satisfies minimum recall >= {MIN_RECALL}"
        )

    return eligible.sort_values(
        by="precision", ascending=False
    ).iloc[0]

# ===============================
# ORCHESTRATOR
# ===============================
def run_model_selection(best_model, X_val, y_val):

    with mlflow.start_run(run_name="Model_Selection"):

        comparison_df = fetch_best_runs()
        mlflow.log_table(comparison_df, "model_comparison.csv")

        champion, challenger, champ_ver, chall_ver = register_and_promote(
            comparison_df
        )

        mlflow.log_param("champion_model", champion.model)
        mlflow.log_metric("champion_val_recall", champion.val_recall)
        mlflow.log_param("champion_run_id", champion.run_id)
        mlflow.log_param("champion_model_version", champ_ver)

        if challenger is not None:
            mlflow.log_param("challenger_model", challenger.model)
            mlflow.log_metric("challenger_val_recall", challenger.val_recall)
            mlflow.log_param("challenger_model_version", chall_ver)

        threshold = tune_threshold(best_model, X_val, y_val)

        mlflow.log_param("decision_threshold", threshold.threshold)
        mlflow.log_metric("threshold_recall", threshold.recall)
        mlflow.log_metric("threshold_precision", threshold.precision)
        mlflow.log_metric("threshold_f1", threshold.f1)

        print(" Model Selection Completed")
        print(f" Champion: {champion.model}")
        print(f" Optimal Threshold: {threshold.threshold}")

        # ====================================================
        # Persist Champion Model (CI / Retraining dependency)
        # ====================================================
        champion_model = mlflow.sklearn.load_model(champion.model_uri)

        BASE_DIR = Path(__file__).resolve().parent
        CHAMPION_PATH = BASE_DIR / "champion.model"

        champion_payload = {
            "model_name": champion.model,
            "model": champion_model,
            "metrics": {
                "val_recall": float(champion.val_recall),
                "val_precision": float(champion.val_precision),
                "val_f1": float(champion.val_f1),
                "val_accuracy": float(champion.val_accuracy),
            },
            "decision_threshold": float(threshold.threshold),
            "run_id": champion.run_id,
            "saved_at_utc": datetime.utcnow().isoformat()
        }

        joblib.dump(champion_payload, CHAMPION_PATH)
        print(f" Champion artifact saved: {CHAMPION_PATH}")

    return {
        "champion_algo": champion.model,
        "champion_run_id": champion.run_id,
        "model_uri": champion.model_uri,
        "decision_threshold": threshold.threshold
    }

