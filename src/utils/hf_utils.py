from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.utils import HfHubHTTPError, RepositoryNotFoundError
import os
from urllib.parse import urlparse
from tqdm import tqdm
from src.utils.defaults import DEFAULT_HEADERS
from google.cloud import storage
from google.cloud.storage import Blob
from typing import Iterator

api = HfApi()


def repo_exists(repo_id: str, repo_type: str = "model") -> bool:
    """
    Check if a HF repo (model, dataset or space) exists.

    :param repo_id: e.g. "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
    :param repo_type: one of "model", "dataset", "space"
    :returns: True if the repo exists, False if 404, re-raises otherwise.
    """
    try:
        # for model repos
        if repo_type == "model":
            api.model_info(repo_id)
        # for dataset repos
        elif repo_type == "dataset":
            api.dataset_info(repo_id)
        # for space repos
        elif repo_type == "space":
            api.space_info(repo_id)
        else:
            raise ValueError(f"Unknown repo_type: {repo_type}")
        return True
    except RepositoryNotFoundError:
        return False
    except HfHubHTTPError as e:
        if e.status_code == 404:
            return False
        raise


def _download_from_gcs(gcs_path: str, save_path: str):
    """Downloads files or directories from Google Cloud Storage."""
    try:
        dest_dir = os.path.join(save_path, os.path.basename(gcs_path.rstrip("/")))

        print(f"Downloading from GCS: {gcs_path} to {dest_dir}")
        storage_client = storage.Client()
        bucket_name, blob_prefix = gcs_path.replace("gs://", "").split("/", 1)
        bucket = storage_client.bucket(bucket_name)
        blobs: Iterator[Blob] = list(bucket.list_blobs(prefix=blob_prefix))

        downloaded = False
        for blob in tqdm(blobs, desc="Downloading from GCS"):
            if blob.name.endswith("/"):
                continue
            destination_file_path = os.path.join(
                dest_dir, os.path.relpath(blob.name, blob_prefix)
            )
            os.makedirs(os.path.dirname(destination_file_path), exist_ok=True)
            if os.path.exists(destination_file_path):
                print(
                    f"File {destination_file_path} already exists, skipping download."
                )
                continue
            blob.download_to_filename(destination_file_path)
            downloaded = True
        if downloaded:
            print(f"Successfully downloaded from GCS: {gcs_path}")
        else:
            print(f"No files found to download from GCS path: {gcs_path}")
    except Exception as e:
        print(
            f"Failed to download from GCS: {gcs_path}. Error: {e}. Please ensure you have authenticated with Google Cloud."
        )


if __name__ == "__main__":
    path = "apex_models/1.3b-standard-1/checkpoint_1070/pytorch_model_fsdp_0"
    local_dir = "/home/tosinkuye/.models"
    _download_from_gcs(path, local_dir)
