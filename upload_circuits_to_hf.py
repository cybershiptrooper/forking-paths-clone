"""Upload results/circuit_discovery/ to a private HuggingFace repository."""

from huggingface_hub import HfApi

REPO_ID = "cybershiptrooper/circuit-discovery-results"
LOCAL_DIR = "results/circuit_discovery"

api = HfApi()

# Create the repo (private, type="dataset" since these are result artifacts)
api.create_repo(repo_id=REPO_ID, private=True, repo_type="dataset", exist_ok=True)

# Upload the entire directory
api.upload_folder(
    folder_path=LOCAL_DIR,
    repo_id=REPO_ID,
    repo_type="dataset",
)

print(f"Uploaded to https://huggingface.co/datasets/{REPO_ID}")
