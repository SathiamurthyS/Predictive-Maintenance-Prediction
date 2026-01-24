from huggingface_hub import HfApi
import os

api = HfApi(token=os.getenv("HF_TOKEN"))

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN missing in environment or HF Space secrets")

api = HfApi(token=HF_TOKEN)


api.upload_folder(
    folder_path="predictive_maintenance/deployment_fe",     # the local folder containing our files
    repo_id="samdurai102024/predictive-maintenance-fe",          # the target repo
    repo_type="space",                      # dataset, model, or space
    path_in_repo="",                          # optional: subfolder path inside the repo
)

api.upload_folder(
    folder_path="predictive_maintenance/deployment_be",     # the local folder containing our files
    repo_id="samdurai102024/predictive-maintenance-be",          # the target repo
    repo_type="space",                      # dataset, model, or space
    path_in_repo="",                          # optional: subfolder path inside the repo
)
