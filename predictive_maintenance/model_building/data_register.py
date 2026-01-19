from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
from huggingface_hub import HfApi, create_repo
import getpass, os

#Try to load HF token from environment (GitHub Actions will set this automatically)
hf_token = os.getenv("HF_TOKEN")

#If not found (e.g., local run / Colab), ask user securely
if hf_token is None:
    print("HF_TOKEN not found in environment. Please enter your Hugging Face token.")
    hf_token = getpass.getpass("Enter HF Token: ")

#Set into environment for the rest of the script
os.environ["HF_TOKEN"] = hf_token

#Enter repo id and repo type
repo_id = "samdurai102024/predictive-maintenance-be"
repo_type = "dataset"

# Initialize API client
api = HfApi(token=os.getenv("HF_TOKEN"))

# Step 1: Check if the space exists
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Space '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Space '{repo_id}' not found. Creating new space...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Space '{repo_id}' created.")

#data upload
api.upload_folder(
    folder_path="predictive_maintenance/data",
    repo_id=repo_id,
    repo_type=repo_type,
)

#Force commit after upload
api.create_commit(
    repo_id=repo_id,
    repo_type=repo_type,
    operations=[],
    commit_message="Force empty commit",
    )
