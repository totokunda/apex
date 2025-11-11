import os
from typing import Iterator, Optional, Callable
from urllib.parse import urlparse
from tenacity import retry, stop_after_attempt, wait_exponential
from src.utils.defaults import DEFAULT_HEADERS, DEFAULT_CONFIG_SAVE_PATH
from logging import Logger
from loguru import logger
import hashlib
from typing import Dict, Any
import tempfile
import shutil

class DownloadMixin:
    logger: Logger = logger

    @staticmethod
    def _get_callback_tqdm(progress_callback: Optional[Callable[[int, Optional[int], Optional[str]], None]] = None):
        """Lazy create CallbackTqdm class when needed"""
        from tqdm import tqdm
        
        class CallbackTqdm(tqdm):
            progress_callback = None

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.progress_callback = progress_callback

            def update(self, n=1):
                res = super().update(n)
                if self.progress_callback:
                    try:
                        total = int(self.total) if self.total is not None else None
                    except Exception:
                        total = None
                    try:
                        self.progress_callback(int(self.n), total, self.desc)
                    except Exception:
                        pass
                return res
        
        return CallbackTqdm

    def _is_url(self, url: str):
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except ValueError:
            return False

    def fetch_config(
        self,
        config_path: str,
        config_save_path: str = DEFAULT_CONFIG_SAVE_PATH,
        return_path: bool = False,
    ):
        path = self._download(config_path, config_save_path)
        if return_path:
            return path
        else:
            return self._load_config_file(path)

    def _save_config(self, config: Dict[str, Any], save_path: str):
        import json
        import yaml
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if save_path.endswith(".json"):
            with open(save_path, "w") as f:
                json.dump(config, f)
        elif save_path.endswith(".yaml"):
            with open(save_path, "w") as f:
                yaml.dump(config, f)
        else:
            raise ValueError(f"Unsupported config file type: {save_path}")
        return save_path

    def download(
        self,
        model_path: str,
        save_path: str,
        progress_callback: Optional[Callable[[int, Optional[int]], None]] = None,
    ):
        return self._download(model_path, save_path, progress_callback)

    @classmethod
    def is_downloaded(cls, model_path: str, save_path: str) -> Optional[str]:
        """Return the expected local path if already downloaded, else None.

        Mirrors destinations used by the download logic without performing any network I/O.
        """

        try:
            def has_file_ending(path: str) -> bool:
                try:
                    if not path:
                        return False
                    parsed = urlparse(path)
                    normalized_path = parsed.path if parsed.scheme else path
                    candidate = normalized_path.rstrip("/")
                    if not candidate:
                        return False
                    filename = os.path.basename(candidate)
                    if "." not in filename:
                        return False
                    lower_name = filename.lower()
                    multipart_suffixes = (".tar.gz", ".tar.bz2", ".tar.xz")
                    if any(lower_name.endswith(sfx) for sfx in multipart_suffixes):
                        return True
                    allowed_extensions = {
                        "json","yaml","yml","toml","ini","cfg","conf",
                        "bin","pt","pth","ckpt","safetensors","gguf","onnx","tflite","h5","hdf5","npz","pb","params","mar",
                        "model","spm","vocab","merges",
                        "zip","tgz","gz","bz2","xz",
                    }
                    ext = lower_name.rsplit(".", 1)[-1]
                    return ext in allowed_extensions
                except Exception:
                    return False

            # Local filesystem path
            if os.path.exists(model_path):
                return model_path

            # Google Drive: destination is not deterministic here
            if "drive.google.com" in model_path:
                return None

            # Google Cloud Storage
            if model_path.startswith("gs://"):
                dest_dir = os.path.join(save_path, os.path.basename(model_path.rstrip("/")))
                return dest_dir if os.path.isdir(dest_dir) and any(os.scandir(dest_dir)) else None

            # AWS S3
            if model_path.startswith("s3://"):
                dest_dir = os.path.join(save_path, os.path.basename(model_path.rstrip("/")))
                return dest_dir if os.path.isdir(dest_dir) and any(os.scandir(dest_dir)) else None

            # Azure Blob Storage
            if "blob.core.windows.net" in model_path:
                parsed = urlparse(model_path)
                if parsed.path:
                    try:
                        _, blob_prefix = parsed.path.strip("/").split("/", 1)
                        dest_dir = os.path.join(save_path, os.path.basename(blob_prefix.rstrip("/")))
                        return dest_dir if os.path.isdir(dest_dir) and any(os.scandir(dest_dir)) else None
                    except ValueError:
                        return None
                return None
            
            parsed_url = urlparse(model_path)
            if parsed_url.scheme and parsed_url.netloc:
                parsed_url = urlparse(model_path)
                relative_path_from_url = parsed_url.path.lstrip("/")
                # Convert to hash of the url
                file_name = hashlib.sha256(model_path.encode('utf-8')).hexdigest()
                base_name = os.path.basename(relative_path_from_url)
                file_path = os.path.join(save_path, f"{file_name}_{base_name}")
                if os.path.isfile(file_path):
                    return file_path if os.path.isfile(file_path) else None
                
            # Hugging Face Hub
            # Decide expected destination based on whether a specific file is referenced
            hf_has_file = has_file_ending(model_path)
            split_path = model_path.split("/")
            if hf_has_file:
                file_name = os.path.basename(model_path)
                file_path = os.path.join(
                    save_path, f"{hashlib.sha256(model_path.encode()).hexdigest()}_{file_name}"
                )
                return file_path if os.path.isfile(file_path) else None

            # If looks like an HF repo path (e.g., namespace/repo[/subfolder/...])
            if len(split_path) >= 2 and "://" not in model_path and not model_path.startswith("/"):
                base_repo = "/".join(split_path[:2])
                base_dir = os.path.join(save_path, base_repo.replace("/", "_"))
                if len(split_path) > 2:
                    sub_dir = os.path.join(base_dir, *split_path[2:])
                    return sub_dir if os.path.isdir(sub_dir) and any(os.scandir(sub_dir)) else None
                return base_dir if os.path.isdir(base_dir) and any(os.scandir(base_dir)) else None


            return None
        except Exception:
            return None

    def _download(
        self,
        model_path: str,
        save_path: str,
        progress_callback: Optional[Callable[[int, Optional[int]], None]] = None,
    ):
        # check if model_path is a local path
        if os.path.exists(model_path):
            return model_path
        elif "drive.google.com" in model_path:
            return self._download_from_google_drive(model_path, save_path, progress_callback)
        elif model_path.startswith("gs://"):
            return self._download_from_gcs(model_path, save_path, progress_callback)
        elif model_path.startswith("s3://"):
            return self._download_from_s3(model_path, save_path, progress_callback)
        elif "blob.core.windows.net" in model_path:
            return self._download_from_azure(model_path, save_path, progress_callback)
        elif self._is_huggingface_repo(model_path):
            return self._download_from_huggingface(model_path, save_path, progress_callback)
        elif self._is_url(model_path):
            return self._download_from_url(model_path, save_path, progress_callback)
        else:
            if hasattr(self, "logger"):
                self.logger.info(f"Skipping download for local path: {model_path}")

        return model_path

    def _is_huggingface_repo(self, model_path: str):
        # if has subfolder in name remove it
        try:
            from huggingface_hub import repo_exists
            namespace, repo_name = model_path.split("/")[:2]
            return repo_exists(f"{namespace}/{repo_name}")
        except Exception as e:
            return False

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=15)
    )
    def _download_from_gcs(
        self,
        gcs_path: str,
        save_path: str,
        progress_callback: Optional[Callable[[int, Optional[int], Optional[str]], None]] = None,
    ):
        try:
            """Downloads files or directories from Google Cloud Storage."""
            from google.cloud import storage
            from google.cloud.storage import Blob
            from tqdm import tqdm
            
            dest_dir = os.path.join(save_path, os.path.basename(gcs_path.rstrip("/")))

            self.logger.info(f"Downloading from GCS: {gcs_path} to {dest_dir}")
            storage_client = storage.Client()
            bucket_name, blob_prefix = gcs_path.replace("gs://", "").split("/", 1)
            bucket = storage_client.bucket(bucket_name)
            blobs: Iterator[Blob] = list(bucket.list_blobs(prefix=blob_prefix))

            # compute total size if available
            total_size_known = True
            total_bytes = 0
            for b in blobs:
                if b.name.endswith("/"):
                    continue
                if b.size is None:
                    total_size_known = False
                else:
                    total_bytes += int(b.size)

            downloaded = False
            bytes_downloaded = 0
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
                    # if size is known, count it
                    if blob.size:
                        bytes_downloaded += int(blob.size)
                        if progress_callback:
                            progress_callback(bytes_downloaded, total_bytes if total_size_known else None, os.path.basename(destination_file_path))
                    continue
                os.makedirs(os.path.dirname(destination_file_path), exist_ok=True)
                # Stream in chunks when possible for time-based progress
                streamed = False
                try:
                    # Prefer streaming via blob.open if available
                    with blob.open("rb") as reader, open(destination_file_path, "wb") as out:
                        while True:
                            chunk = reader.read(8 * 1024 * 1024)
                            if not chunk:
                                break
                            out.write(chunk)
                            bytes_downloaded += len(chunk)
                            if progress_callback:
                                progress_callback(bytes_downloaded, total_bytes if total_size_known else None, os.path.basename(destination_file_path))
                    streamed = True
                except Exception:
                    streamed = False

                if not streamed:
                    blob.download_to_filename(destination_file_path)
                    if blob.size:
                        bytes_downloaded += int(blob.size)
                    if progress_callback:
                        progress_callback(bytes_downloaded, total_bytes if total_size_known else None, os.path.basename(destination_file_path))
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
    def _download_from_s3(
        self,
        s3_path: str,
        save_path: str,
        progress_callback: Optional[Callable[[int, Optional[int], Optional[str]], None]] = None,
    ):
        """Downloads files or directories from AWS S3."""
        try:
            import boto3
            from botocore.exceptions import NoCredentialsError
            
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
            pages = list(paginator.paginate(Bucket=bucket_name, Prefix=s3_prefix))

            # compute total size
            total_bytes = 0
            for page in pages:
                if "Contents" in page:
                    for obj in page["Contents"]:
                        if obj["Key"].endswith("/"):
                            continue
                        total_bytes += int(obj.get("Size", 0))

            downloaded = False
            bytes_downloaded = 0
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
                            bytes_downloaded += int(obj.get("Size", 0))
                            if progress_callback:
                                progress_callback(bytes_downloaded, total_bytes or None, os.path.basename(dest_path))
                            continue
                        if not os.path.exists(os.path.dirname(dest_path)):
                            os.makedirs(os.path.dirname(dest_path))
                        if not key.endswith("/"):
                            # Use TransferConfig with callback for streaming progress
                            cb_accum = {"n": 0}
                            def _cb(bytes_amount):
                                cb_accum["n"] += int(bytes_amount)
                                if progress_callback:
                                    progress_callback(bytes_downloaded + cb_accum["n"], total_bytes or None, os.path.basename(dest_path))
                            s3_client.download_file(
                                bucket_name,
                                key,
                                dest_path,
                                Callback=_cb,
                            )
                            bytes_downloaded += cb_accum["n"]
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
    def _download_from_azure(
        self,
        azure_url: str,
        save_path: str,
        progress_callback: Optional[Callable[[int, Optional[int], Optional[str]], None]] = None,
    ):
        """Downloads files or blobs from Azure Blob Storage."""
        try:
            from azure.storage.blob import BlobServiceClient
            
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

            blob_list = list(container_client.list_blobs(name_starts_with=blob_prefix))
            total_bytes = sum(int(b.size or 0) for b in blob_list if not b.name.endswith("/"))
            bytes_downloaded = 0
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
                    bytes_downloaded += int(blob.size or 0)
                    if progress_callback:
                        progress_callback(bytes_downloaded, total_bytes or None, os.path.basename(file_path))
                    continue

                blob_client = blob_service_client.get_blob_client(
                    container=container_name, blob=blob.name
                )
                with tempfile.NamedTemporaryFile() as tmp_file:
                    downloader = blob_client.download_blob()
                    for chunk in downloader.chunks():
                        if not chunk:
                            continue
                        tmp_file.write(chunk)
                        bytes_downloaded += len(chunk)
                        if progress_callback:
                            progress_callback(bytes_downloaded, total_bytes or None, os.path.basename(file_path))
                    shutil.move(tmp_file.name, file_path)
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

    def _has_file_ending(self, path: str):
        try:
            if not path:
                return False

            # Normalize path component (strip query/fragment for URLs)
            parsed = urlparse(path)
            # For URLs, use parsed.path; for local/cloud-style URIs without netloc, fallback to original
            normalized_path = parsed.path if parsed.scheme else path

            candidate = normalized_path.rstrip("/")
            if not candidate:
                return False

            filename = os.path.basename(candidate)
            if "." not in filename:
                return False

            lower_name = filename.lower()

            # Multi-part and specific endings first
            multipart_suffixes = (
                ".tar.gz",
                ".tar.bz2",
                ".tar.xz",
            )
            if any(lower_name.endswith(sfx) for sfx in multipart_suffixes):
                return True

            # Typical config and model weight endings (single-segment)
            allowed_extensions = {
                # Configs
                "json",
                "yaml",
                "yml",
                "toml",
                "ini",
                "cfg",
                "conf",
                # Model weights / artifacts
                "bin",
                "pt",
                "pth",
                "ckpt",
                "safetensors",
                "onnx",
                "tflite",
                "h5",
                "hdf5",
                "npz",
                "pb",
                "params",
                "mar",
                "gguf",
                # Tokenizer/vocab related (often shipped with models)
                "model",
                "spm",
                "vocab",
                "merges",
                # Archives / compressed
                "zip",
                "tgz",
                "gz",
                "bz2",
                "xz",
            }

            ext = lower_name.rsplit(".", 1)[-1]
            return ext in allowed_extensions
        except Exception:
            return False

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=15)
    )
    def _download_from_huggingface(
        self,
        repo_id: str,
        save_path: str,
        progress_callback: Optional[Callable[[int, Optional[int], Optional[str]], None]] = None,
    ):
        """Downloads a repository from the Hugging Face Hub."""
        try:
            import huggingface_hub
            import shutil

            if hasattr(self, "logger"):
                self.logger.info(f"Downloading from Hugging Face Hub: {repo_id}")

            split_path = repo_id.split("/")

            if self._has_file_ending(repo_id):
                # fetch the specific file
                self.logger.info(
                    f"Downloading specific file from Hugging Face Hub: {repo_id}"
                )
                file_name = os.path.basename(repo_id)
                file_path = (
                    f"{hashlib.sha256(repo_id.encode()).hexdigest()}_{file_name}"
                )
                # hash the des
                file_path = os.path.join(save_path, file_path)
                if os.path.exists(file_path):
                    self.logger.info(
                        f"File {file_path} already exists, skipping download."
                    )
                    return file_path
                repo_id = "/".join(
                    split_path if len(split_path) <= 2 else split_path[:2]
                )
                subfolder = (
                    f"{'/'.join(split_path[2:-1])}" if len(split_path) > 2 else None
                )
                # configure callback-enabled tqdm if patch is available
                try:
                    callback_tqdm = self._get_callback_tqdm(progress_callback)
                    with tempfile.TemporaryDirectory() as tmp_dir:
                        curr_save_path = huggingface_hub.hf_hub_download(
                            repo_id,
                            file_name,
                            local_dir=tmp_dir,
                            subfolder=subfolder,
                            tqdm_class=callback_tqdm,
                        )
                        shutil.move(curr_save_path, file_path)
                        self.logger.info(f"Successfully downloaded specific file from Hugging Face Hub: {repo_id}")
                        return file_path
                except TypeError as e:
                    self.logger.error(f"Failed to download from Hugging Face Hub: {repo_id}. Error: {e}")
                    curr_save_path = huggingface_hub.hf_hub_download(
                        repo_id, file_name, local_dir=save_path, subfolder=subfolder
                    )
                # move the file to the correct path
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                shutil.move(curr_save_path, file_path)

                self.logger.info(
                    f"Successfully downloaded specific file from Hugging Face Hub: {repo_id}"
                )
                return file_path

            subfolder = (
                [f"{'/'.join(split_path[2:])}/*"] if len(split_path) > 2 else None
            )
            repo_id = "/".join(split_path if len(split_path) <= 2 else split_path[:2])

            dest_path = os.path.join(save_path, repo_id.replace("/", "_"))
            # check if dest_path exists and is not empty
            
            if os.path.exists(dest_path) and os.listdir(dest_path):
                if not subfolder:
                    self.logger.info(f"Directory {dest_path} already exists and is not empty, skipping download.")
                    return dest_path
                else:
                    # check if subfolder exists and is not empty
                    subfolder_path = os.path.join(dest_path, *subfolder[0].split("/")[:-1])
                    if os.path.exists(subfolder_path) and os.listdir(subfolder_path):
                        self.logger.info(f"Directory {subfolder_path} already exists and is not empty, skipping download.")
                        return subfolder_path
            # pass custom tqdm if supported by version
            # We use a temp directory to download the repository
            with tempfile.TemporaryDirectory() as tmp_dir:
                
                callback_tqdm = self._get_callback_tqdm(progress_callback)
                tmp_download_path = huggingface_hub.snapshot_download(
                        repo_id,
                        local_dir=tmp_dir,
                        local_dir_use_symlinks=False,
                        allow_patterns=subfolder,
                        tqdm_class=callback_tqdm,
                )
               
                if subfolder:
                    tmp_download_path = os.path.join(tmp_download_path, *subfolder[0].split("/")[:-1])
                shutil.move(tmp_download_path, dest_path)
                if hasattr(self, "logger"):
                    self.logger.info(
                        f"Successfully downloaded from Hugging Face Hub: {repo_id}"
                    )
                if progress_callback:
                    try:
                        total = 0
                        for root, _, files in os.walk(dest_path):
                            for f in files:
                                fp = os.path.join(root, f)
                                try:
                                    total += os.path.getsize(fp)
                                except OSError:
                                    pass
                        progress_callback(total, total)
                    except Exception:
                        progress_callback(0, None)
            return dest_path
        except Exception as e:
            if hasattr(self, "logger"):
                self.logger.error(
                    f"Failed to download from Hugging Face Hub: {repo_id}. Error: {e}"
                )

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=15)
    )
    def _download_from_google_drive(
        self,
        url: str,
        save_path: str,
        progress_callback: Optional[Callable[[int, Optional[int], Optional[str]], None]] = None,
    ):
        try:
            from gdown import download
            with tempfile.NamedTemporaryFile() as tmp_file:
                # gdown does not expose streaming progress easily; call and then best-effort size report
                path = download(url, tmp_file.name, user_agent=DEFAULT_HEADERS["User-Agent"])
                if progress_callback and path and os.path.exists(path):
                    try:
                        size = os.path.getsize(path)
                        progress_callback(size, size, os.path.basename(path))
                    except Exception:
                        progress_callback(0, None, None)
                shutil.move(tmp_file.name, save_path)
                return save_path
        except Exception as e:
            self.logger.error(
                f"Failed to download from Google Drive: {url}. Error: {e}"
            )

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=15)
    )
    def _download_from_url(
        self,
        url: str,
        save_path: str,
        progress_callback: Optional[Callable[[int, Optional[int], Optional[str]], None]] = None,
    ):
        try:
            """Downloads a single file from a URL."""
            import requests
            from tqdm import tqdm
            
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                parsed_url = urlparse(url)
                relative_path_from_url = parsed_url.path.lstrip("/")
                # Convert to hash of the url
                file_name = hashlib.sha256(url.encode('utf-8')).hexdigest()
                base_name = os.path.basename(relative_path_from_url)
                file_path = os.path.join(save_path, f"{file_name}_{base_name}")
                
                if os.path.exists(file_path):
                    self.logger.info(f"File {file_path} already exists, skipping download.")
                    return
                self.logger.info(f"Downloading {file_name} from: {url}")
                response = requests.get(
                    url, timeout=10, verify=False, headers=DEFAULT_HEADERS, stream=True
                )
                response.raise_for_status()
                
                total_size = int(response.headers.get("content-length", 0))

                downloaded_so_far = 0
                with tqdm(
                    desc=os.path.basename(file_path),
                    total=total_size,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                ) as bar:
                    for chunk in response.iter_content(chunk_size=10000000):
                        if chunk:
                            tmp_file.write(chunk)
                            chunk_len = len(chunk)
                            bar.update(chunk_len)
                            downloaded_so_far += chunk_len
                            if progress_callback:
                                progress_callback(downloaded_so_far, total_size or None, os.path.basename(file_path))
                
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                shutil.move(tmp_file.name, file_path)
                self.logger.info(f"Successfully downloaded {file_name} to {file_path}")
        except Exception as e:
            self.logger.error(f"Failed to download from URL: {url}. Error: {e}")
        finally:
            return file_path
