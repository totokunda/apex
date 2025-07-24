import os
from typing import Iterator
from urllib.parse import urlparse
import requests
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_exponential
from src.utils.defaults import DEFAULT_HEADERS
from logging import Logger
from huggingface_hub import repo_exists
from google.cloud import storage
import boto3
from azure.storage.blob import BlobServiceClient
from botocore.exceptions import NoCredentialsError
import huggingface_hub
from google.cloud.storage import Blob
from loguru import logger


class DownloadMixin:
    logger: Logger = logger

    def _is_url(self, url: str):
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except ValueError:
            return False

    def _download(self, model_path: str, save_path: str):
        if model_path.startswith("gs://"):
            return self._download_from_gcs(model_path, save_path)
        elif model_path.startswith("s3://"):
            return self._download_from_s3(model_path, save_path)
        elif "blob.core.windows.net" in model_path:
            return self._download_from_azure(model_path, save_path)
        elif self._is_huggingface_repo(model_path):
            return self._download_from_huggingface(model_path, save_path)
        elif self._is_url(model_path):
            return self._download_from_url(model_path, save_path)
        else:
            if hasattr(self, "logger"):
                self.logger.info(f"Skipping download for local path: {model_path}")

        return model_path

    def _is_huggingface_repo(self, model_path: str):
        # if has subfolder in name remove it
        try:
            namespace, repo_name = model_path.split("/")[:2]
            return repo_exists(f"{namespace}/{repo_name}")
        except Exception as e:
            return False

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=15)
    )
    def _download_from_gcs(self, gcs_path: str, save_path: str):
        try:
            """Downloads files or directories from Google Cloud Storage."""
            dest_dir = os.path.join(save_path, os.path.basename(gcs_path.rstrip("/")))

            self.logger.info(f"Downloading from GCS: {gcs_path} to {dest_dir}")
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
                if os.path.exists(destination_file_path):
                    self.logger.info(
                        f"File {destination_file_path} already exists, skipping download."
                    )
                    continue
                os.makedirs(os.path.dirname(destination_file_path), exist_ok=True)
                blob.download_to_filename(destination_file_path)
                downloaded = True
            if downloaded:
                self.logger.info(f"Successfully downloaded from GCS: {gcs_path}")
            else:
                self.logger.warning(
                    f"No files found to download from GCS path: {gcs_path}"
                )
        except Exception as e:
            self.logger.error(
                f"Failed to download from GCS: {gcs_path}. Error: {e}. Please ensure you have authenticated with Google Cloud."
            )
        finally:
            return dest_dir

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=15)
    )
    def _download_from_s3(self, s3_path: str, save_path: str):
        """Downloads files or directories from AWS S3."""
        try:
            dest_dir = os.path.join(save_path, os.path.basename(s3_path.rstrip("/")))
            if os.path.isdir(dest_dir) and os.listdir(dest_dir):
                self.logger.info(
                    f"Directory {dest_dir} already exists and is not empty, skipping download."
                )
                return

            self.logger.info(f"Downloading from S3: {s3_path} to {dest_dir}")
            s3_client = boto3.client("s3")
            bucket_name, s3_prefix = s3_path.replace("s3://", "").split("/", 1)

            paginator = s3_client.get_paginator("list_objects_v2")
            pages = paginator.paginate(Bucket=bucket_name, Prefix=s3_prefix)

            downloaded = False
            for page in pages:
                if "Contents" in page:
                    for obj in page["Contents"]:
                        key = obj["Key"]
                        dest_path = os.path.join(
                            dest_dir, os.path.relpath(key, s3_prefix)
                        )
                        if os.path.exists(dest_path):
                            self.logger.info(
                                f"File {dest_path} already exists, skipping download."
                            )
                            continue
                        if not os.path.exists(os.path.dirname(dest_path)):
                            os.makedirs(os.path.dirname(dest_path))
                        if not key.endswith("/"):
                            s3_client.download_file(bucket_name, key, dest_path)
                            downloaded = True

            if downloaded:
                self.logger.info(f"Successfully downloaded from S3: {s3_path}")
            else:
                self.logger.warning(
                    f"No files found to download from S3 path: {s3_path}"
                )
        except NoCredentialsError:
            self.logger.error(
                "AWS credentials not found. Please configure your credentials."
            )
        except Exception as e:
            self.logger.error(f"Failed to download from S3: {s3_path}. Error: {e}")
        finally:
            return dest_dir

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=15)
    )
    def _download_from_azure(self, azure_url: str, save_path: str):
        """Downloads files or blobs from Azure Blob Storage."""
        try:
            url_parts = urlparse(azure_url)
            if not url_parts.path:
                self.logger.error(f"Invalid Azure URL: {azure_url}. Path is missing.")
                return

            container_name, blob_prefix = url_parts.path.strip("/").split("/", 1)
            dest_dir = os.path.join(
                save_path, os.path.basename(blob_prefix.rstrip("/"))
            )
            self.logger.info(f"Downloading from Azure: {azure_url} to {dest_dir}")

            account_url = f"{url_parts.scheme}://{url_parts.netloc}"

            blob_service_client = BlobServiceClient(account_url=account_url)
            container_client = blob_service_client.get_container_client(container_name)

            blob_list = container_client.list_blobs(name_starts_with=blob_prefix)
            downloaded = False
            for blob in blob_list:
                if blob.name.endswith("/"):
                    continue
                file_path = os.path.join(
                    dest_dir, os.path.relpath(blob.name, blob_prefix)
                )
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                if os.path.exists(file_path):
                    self.logger.info(
                        f"File {file_path} already exists, skipping download."
                    )
                    continue

                blob_client = blob_service_client.get_blob_client(
                    container=container_name, blob=blob.name
                )
                with open(file_path, "wb") as download_file:
                    download_file.write(blob_client.download_blob().readall())
                downloaded = True

            if downloaded:
                self.logger.info(f"Successfully downloaded from Azure: {azure_url}")
            else:
                self.logger.warning(
                    f"No files found to download from Azure path: {azure_url}"
                )
        except Exception as e:
            self.logger.error(
                f"Failed to download from Azure: {azure_url}. Error: {e}. Please ensure you have authenticated with Azure."
            )
        finally:
            return dest_dir

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=15)
    )
    def _download_from_huggingface(self, repo_id: str, save_path: str):
        """Downloads a repository from the Hugging Face Hub."""
        try:
            dest_path = os.path.join(save_path, repo_id.replace("/", "_"))
            if hasattr(self, "logger"):
                self.logger.info(f"Downloading from Hugging Face Hub: {repo_id}")
            split_path = repo_id.split("/")
            subfolder = (
                [f"{'/'.join(split_path[2:])}/*"] if len(split_path) > 2 else None
            )
            repo_id = "/".join(split_path if len(split_path) <= 2 else split_path[:2])

            dest_path = huggingface_hub.snapshot_download(
                repo_id,
                local_dir=dest_path,
                local_dir_use_symlinks=False,
                allow_patterns=subfolder,
            )
            if subfolder:
                dest_path = os.path.join(dest_path, subfolder[0].split("/")[0])
            if hasattr(self, "logger"):
                self.logger.info(
                    f"Successfully downloaded from Hugging Face Hub: {repo_id}"
                )
        except Exception as e:
            if hasattr(self, "logger"):
                self.logger.error(
                    f"Failed to download from Hugging Face Hub: {repo_id}. Error: {e}"
                )
        finally:
            return dest_path

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=15)
    )
    def _download_from_url(self, url: str, save_path: str):
        try:
            """Downloads a single file from a URL."""
            parsed_url = urlparse(url)

            relative_path_from_url = parsed_url.path.lstrip("/")
            file_name = os.path.basename(relative_path_from_url)

            # create all subfolders needed
            file_path = os.path.join(save_path, relative_path_from_url)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            if os.path.exists(file_path):
                self.logger.info(f"File {file_path} already exists, skipping download.")
                return
            self.logger.info(f"Downloading {file_name} from: {url}")
            response = requests.get(
                url, timeout=10, verify=False, headers=DEFAULT_HEADERS, stream=True
            )
            response.raise_for_status()

            total_size = int(response.headers.get("content-length", 0))

            with open(file_path, "wb") as f, tqdm(
                desc=file_name,
                total=total_size,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        bar.update(len(chunk))
            self.logger.info(f"Successfully downloaded {file_name} to {file_path}")
        except Exception as e:
            self.logger.error(f"Failed to download from URL: {url}. Error: {e}")
        finally:
            return file_path
