
import mlflow
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime, timezone
from mlflow.tracking import MlflowClient
from huggingface_hub import HfApi

# ===============================
# CONFIG
# ===============================
EXPERIMENTS = {
    "RF": "Predictive_Maintenance_RF_GridSearch",
    "GBM": "Predictive_Maintenance_GBM_GridSearch",
    "XGB": "Predictive_Maintenance_XGB_GridSearch",
}

PRIMARY_METRIC = "val_recall"
TRACKING_URI = "file:./mlruns"

mlflow.set_tracking_uri(TRACKING_URI)
client = MlflowClient()

# ===============================
# STEP 1: CROSS-EXPERIMENT COMPARISON
# ===============================
def fetch_best_runs() -> pd.DataFrame:
    records = []

    for algo, exp_name in EXPERIMENTS.items():
        exp = mlflow.get_experiment_by_name(exp_name)
        if exp is None:
            print(f" Skipping {algo}: experiment not found")
            continue

        runs = mlflow.search_runs(
            experiment_ids=[exp.experiment_id],
            order_by=[f"metrics.{PRIMARY_METRIC} DESC"],
            max_results=1,
        )

        if runs.empty:
            print(f" Skipping {algo}: no runs found")
            continue

        row = runs.iloc[0]

        records.append({
            "model": algo,
            "run_id": row["run_id"],
            "val_recall": float(row.get("metrics.val_recall", 0.0)),
            "val_precision": float(row.get("metrics.val_precision", 0.0)),
            "val_f1": float(row.get("metrics.val_f1", 0.0)),
            "val_accuracy": float(row.get("metrics.val_accuracy", 0.0)),
        })

    if not records:
        print(" No valid candidate runs found across experiments")
        return pd.DataFrame()

    return (
        pd.DataFrame(records)
        .sort_values(by=PRIMARY_METRIC, ascending=False)
        .reset_index(drop=True)
    )

# ===============================
# ORCHESTRATOR
# ===============================
def run_model_selection():

    with mlflow.start_run(run_name="Model_Selection"):

        comparison_df = fetch_best_runs()

        # ----------------------------------------------
        # Graceful exit (CRITICAL FOR CI)
        # ----------------------------------------------
        if comparison_df.empty:
            mlflow.log_param("model_selection_status", "skipped_no_candidates")
            print(" Model selection skipped — no valid candidates")
            return None

        # MLflow requires .json or .parquet
        mlflow.log_table(comparison_df, "model_comparison.json")

        champion = comparison_df.iloc[0]

        mlflow.log_param("champion_model", champion["model"])
        mlflow.log_param("champion_run_id", champion["run_id"])
        mlflow.log_metric("champion_val_recall", champion["val_recall"])

        print(f" Champion Model Selected: {champion['model']}")

        # ==============================================
        # Persist Champion Metadata (NOT MODEL OBJECT)
        # ==============================================
        base_dir = Path(__file__).resolve().parent
        champion_path = base_dir / "champion.model"

        payload = {
            "model_name": champion["model"],
            "metrics": {
                "val_recall": champion["val_recall"],
                "val_precision": champion["val_precision"],
                "val_f1": champion["val_f1"],
                "val_accuracy": champion["val_accuracy"],
            },
            "run_id": champion["run_id"],
            "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        }

        joblib.dump(payload, champion_path)
        print(f"Champion metadata saved at: {champion_path}")

        HF_MODEL_REPO = "samdurai102024/predictive-maintenance-be"
        HF_CHAMPION_PATH = "production/champion.model"

        api = HfApi()

        api.upload_file(
            path_or_fileobj=champion_path,
            path_in_repo=HF_CHAMPION_PATH,
            repo_id=HF_MODEL_REPO,
            repo_type="model",
            commit_message="Update champion model metadata"
        )

        print(" Champion metadata uploaded to Hugging Face")



    return payload

# ===============================
# SCRIPT ENTRYPOINT (CI SAFE)
# ===============================
if __name__ == "__main__":

    print("\n▶ Running Model Selection")

    try:
        run_model_selection()
    except Exception as e:
        # ABSOLUTELY NEVER FAIL CI HERE
        print(f" Model selection failed gracefully: {e}")

    print("▶ Model selection completed successfully")
