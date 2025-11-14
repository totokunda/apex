import os
from typing import Iterator, Optional, Callable
from urllib.parse import urlparse, parse_qs
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
    def _stable_url_key(url: str) -> str:
        """Return a stable key for a URL that ignores volatile query/fragment signing parts.
        
        Uses scheme://netloc/path when possible; falls back to the original string on parse errors.
        """
        try:
            parsed = urlparse(url)
            if parsed.scheme and parsed.netloc:
                return f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
            return url
        except Exception:
            return url

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
    
    def _requests_verify(self) -> bool:
        """Return whether HTTPS certificate verification should be enabled. Defaults to True.
        
        Can be disabled by setting env var APEX_REQUESTS_VERIFY to 'false', '0', 'no', or 'off'.
        """
        try:
            flag = os.environ.get("APEX_REQUESTS_VERIFY")
            if flag is None:
                return True
            return str(flag).strip().lower() not in ("0", "false", "no", "off")
        except Exception:
            return True

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
            def compute_hash(s: str) -> str:
                try:
                    return hashlib.sha256(s.encode("utf-8")).hexdigest()
                except Exception:
                    return hashlib.sha256(repr(s).encode("utf-8")).hexdigest()

            def is_complete_file(path: str) -> bool:
                try:
                    if not os.path.isfile(path):
                        return False
                    if path.endswith(".part"):
                        return False
                    # Must not have an associated partial file and should be non-empty
                    if os.path.exists(f"{path}.part"):
                        return False
                    return os.path.getsize(path) > 0
                except Exception:
                    return False

            def dir_is_complete(path: str) -> bool:
                """A directory is considered complete if:
                - It contains at least one file (recursively)
                - No '*.part' files are present anywhere under it
                """
                try:
                    found_any_file = False
                    for root, _dirs, files in os.walk(path):
                        for name in files:
                            if name.endswith(".part"):
                                return False
                            # Count only real files (zero-length may be legitimate, so don't require > 0 here)
                            found_any_file = True or found_any_file
                    return found_any_file
                except Exception:
                    return False

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
                        # Common text/docs
                        "md","txt",
                    }
                    ext = lower_name.rsplit(".", 1)[-1]
                    return ext in allowed_extensions
                except Exception:
                    return False

            # Local filesystem path
            if os.path.exists(model_path):
                if os.path.isfile(model_path):
                    return model_path if is_complete_file(model_path) else None
                if os.path.isdir(model_path):
                    return model_path if dir_is_complete(model_path) else None
                return None

            # Google Drive: destination is not deterministic here
            if "drive.google.com" in model_path:
                file_hashes = [hashlib.sha256(model_path.encode('utf-8')).hexdigest()]
                try:
                    q = parse_qs(urlparse(model_path).query)
                    file_id = (q.get("id") or [None])[0]
                    if file_id:
                        file_hashes.append(hashlib.sha256(f"gdrive:{file_id}".encode("utf-8")).hexdigest())
                except Exception:
                    pass
                try:
                    for name in os.listdir(save_path):
                        if any(name.startswith(f"{h}_") for h in file_hashes):
                            candidate_path = os.path.join(save_path, name)
                            if is_complete_file(candidate_path):
                                return candidate_path
                except Exception:
                    pass
                return None

            # Google Cloud Storage
            if model_path.startswith("gs://"):
                dest_dir = os.path.join(save_path, os.path.basename(model_path.rstrip("/")))
                return dest_dir if os.path.isdir(dest_dir) and dir_is_complete(dest_dir) else None

            # AWS S3
            if model_path.startswith("s3://"):
                dest_dir = os.path.join(save_path, os.path.basename(model_path.rstrip("/")))
                return dest_dir if os.path.isdir(dest_dir) and dir_is_complete(dest_dir) else None

            # Azure Blob Storage
            if "blob.core.windows.net" in model_path:
                parsed = urlparse(model_path)
                if parsed.path:
                    try:
                        _, blob_prefix = parsed.path.strip("/").split("/", 1)
                        dest_dir = os.path.join(save_path, os.path.basename(blob_prefix.rstrip("/")))
                        return dest_dir if os.path.isdir(dest_dir) and dir_is_complete(dest_dir) else None
                    except ValueError:
                        return None
                return None
            
            parsed_url = urlparse(model_path)
            if parsed_url.scheme and parsed_url.netloc:
                relative_path_from_url = parsed_url.path.lstrip("/")
                # Prefer a stable key that ignores volatile query string
                stable_key = cls._stable_url_key(model_path)
                # Consider both old (full URL hash) and new (stable key hash) for backwards compatibility
                candidate_hashes = [
                    compute_hash(model_path),
                    compute_hash(stable_key),
                ]
                base_name = os.path.basename(relative_path_from_url) or "download"
                for h in candidate_hashes:
                    exact_path = os.path.join(save_path, f"{h}_{base_name}")
                    if is_complete_file(exact_path):
                        return exact_path
                # Fallback: search by either hash prefix (handles Content-Disposition filename differences)
                try:
                    for name in os.listdir(save_path):
                        if any(name.startswith(f"{h}_") for h in candidate_hashes):
                            candidate_path = os.path.join(save_path, name)
                            if is_complete_file(candidate_path):
                                return candidate_path
                except Exception:
                    pass
                return None
                
            # Hugging Face Hub
            # Decide expected destination based on whether a specific file is referenced
            hf_has_file = has_file_ending(model_path)
            split_path = model_path.split("/")
            if hf_has_file:
                file_name = os.path.basename(model_path)
                file_path = os.path.join(
                    save_path, f"{hashlib.sha256(model_path.encode()).hexdigest()}_{file_name}"
                )
                return file_path if is_complete_file(file_path) else None

            # If looks like an HF repo path (e.g., namespace/repo[/subfolder/...])
            if len(split_path) >= 2 and "://" not in model_path and not model_path.startswith("/"):
                base_repo = "/".join(split_path[:2])
                base_dir = os.path.join(save_path, base_repo.replace("/", "_"))
                if len(split_path) > 2:
                    sub_dir = os.path.join(base_dir, *split_path[2:])
                    return sub_dir if os.path.isdir(sub_dir) and dir_is_complete(sub_dir) else None
                return base_dir if os.path.isdir(base_dir) and dir_is_complete(base_dir) else None


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
            from huggingface_hub import repo_exists, get_token
            namespace, repo_name = model_path.split("/")[:2]
            token = get_token()
            logger.info(f"Checking if repo exists: {namespace}/{repo_name}")
            logger.info(f"Token: {token}")
            logger.info(f"Repo exists: {repo_exists(f'{namespace}/{repo_name}', token=token)}")
            return repo_exists(f"{namespace}/{repo_name}", token=token)
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
            from datetime import timedelta
            
            dest_dir = os.path.join(save_path, os.path.basename(gcs_path.rstrip("/")))

            self.logger.info(f"Downloading from GCS: {gcs_path} to {dest_dir}")
            # Allow specifying a credentials.json path via environment variable for restricted access
            # Prefer APEX_GCS_CREDENTIALS, fallback to GCS_CREDENTIALS_JSON; otherwise use default client behavior.
            cred_path = os.environ.get("APEX_GCS_CREDENTIALS") or os.environ.get("GCS_CREDENTIALS_JSON")
            if cred_path:
                try:
                    if not os.path.isfile(cred_path):
                        raise FileNotFoundError(f"GCS credentials file not found at {cred_path}")
                    storage_client = storage.Client.from_service_account_json(cred_path)
                    self.logger.info("Using GCS credentials from environment variable")
                except Exception as e:
                    self.logger.warning(f"Failed to initialize GCS client with provided credentials file '{cred_path}': {e}. Falling back to default credentials.")
                    storage_client = storage.Client()
            else:
                # storage.Client() will pick up GOOGLE_APPLICATION_CREDENTIALS or default ADC if configured
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
                # Build relative path robustly; avoid '.' when blob.name equals the prefix
                prefix_norm = blob_prefix.rstrip("/")
                if blob.name == prefix_norm:
                    relative_path = os.path.basename(blob.name)
                elif blob.name.startswith(prefix_norm + "/"):
                    relative_path = blob.name[len(prefix_norm) + 1 :]
                else:
                    rp = os.path.relpath(blob.name, prefix_norm)
                    relative_path = rp if rp not in (".", "") else os.path.basename(blob.name)
                destination_file_path = os.path.join(dest_dir, relative_path)

                file_label = os.path.basename(destination_file_path)
                if os.path.exists(destination_file_path):
                    self.logger.info(
                        f"File {destination_file_path} already exists, skipping download."
                    )
                    # if size is known, count it; otherwise compute from local file
                    size_to_add = None
                    if blob.size:
                        size_to_add = int(blob.size)
                    else:
                        try:
                            size_to_add = os.path.getsize(destination_file_path)
                        except Exception:
                            size_to_add = None
                    # Report per-file completion, not aggregate
                    if size_to_add and progress_callback:
                        progress_callback(size_to_add, size_to_add, file_label)
                    if size_to_add:
                        bytes_downloaded += size_to_add
                    continue

                # Resolve a URL for the blob: prefer a short-lived signed URL, fallback to public URL
                try:
                    signed_url = blob.generate_signed_url(
                        version="v4",
                        expiration=timedelta(hours=1),
                        method="GET",
                    )
                except Exception:
                    try:
                        signed_url = blob.public_url
                    except Exception:
                        signed_url = None

                if not signed_url:
                    # Fallback to SDK download if URL cannot be resolved
                    os.makedirs(os.path.dirname(destination_file_path), exist_ok=True)
                    blob.download_to_filename(destination_file_path)
                    size_to_add = None
                    if blob.size:
                        size_to_add = int(blob.size)
                    else:
                        try:
                            size_to_add = os.path.getsize(destination_file_path)
                        except Exception:
                            size_to_add = None
                    # Report per-file completion, not aggregate
                    if progress_callback and size_to_add:
                        progress_callback(size_to_add, size_to_add, file_label)
                    if size_to_add:
                        bytes_downloaded += size_to_add
                    downloaded = True
                    continue

                # Use URL-based downloader for better progress reporting
                def _per_file_cb(n: int, _total: Optional[int], _label: Optional[str] = None):
                    if progress_callback:
                        # Pass through per-file bytes and per-file total
                        progress_callback(int(n), _total, _label or file_label)

                # Download directly to destination with resume support using deterministic dest_path
                os.makedirs(os.path.dirname(destination_file_path), exist_ok=True)
                self._download_from_url(
                    url=signed_url,
                    save_path=os.path.dirname(destination_file_path),
                    progress_callback=_per_file_cb,
                    filename=file_label,
                    # ensure resume across different signed URLs
                    dest_path=destination_file_path,
                )

                # Update aggregated progress after file completes
                size_to_add = None
                if blob.size:
                    size_to_add = int(blob.size)
                else:
                    try:
                        size_to_add = os.path.getsize(destination_file_path)
                    except Exception:
                        size_to_add = None
                if size_to_add:
                    bytes_downloaded += size_to_add
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
                            file_size = int(obj.get("Size", 0)) if obj.get("Size") is not None else None
                            if progress_callback and file_size:
                                # Report per-file completion
                                progress_callback(file_size, file_size, os.path.basename(dest_path))
                            if file_size:
                                bytes_downloaded += file_size
                            continue
                        if not os.path.exists(os.path.dirname(dest_path)):
                            os.makedirs(os.path.dirname(dest_path))
                        if not key.endswith("/"):
                            # Use TransferConfig with callback for streaming progress
                            cb_accum = {"n": 0}
                            file_total = int(obj.get("Size", 0)) if obj.get("Size") is not None else None
                            def _cb(bytes_amount):
                                cb_accum["n"] += int(bytes_amount)
                                if progress_callback:
                                    # Report per-file progress with per-file total
                                    progress_callback(cb_accum["n"], file_total, os.path.basename(dest_path))
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
            sas_suffix = f"?{url_parts.query}" if url_parts.query else ""
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
                    per_file_size = int(blob.size or 0) if blob.size else None
                    if progress_callback and per_file_size:
                        # Report per-file completion
                        progress_callback(per_file_size, per_file_size, os.path.basename(file_path))
                    if per_file_size:
                        bytes_downloaded += per_file_size
                    continue

                # Build per-blob URL (append SAS from original URL if present) and delegate to URL downloader
                blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob.name)
                blob_url = f"{blob_client.url}{sas_suffix}"

                # Offset per-file progress into aggregate total
                start_offset = bytes_downloaded

                def _agg_progress(per_file_downloaded: int, _per_file_total: Optional[int], filename: Optional[str]):
                    if progress_callback:
                        try:
                            # Report per-file progress and per-file total (prefer per-call total; fallback to blob.size)
                            total_for_file = _per_file_total if _per_file_total is not None else (int(blob.size or 0) or None)
                            progress_callback(int(per_file_downloaded or 0), total_for_file, filename or os.path.basename(file_path))
                        except Exception:
                            pass

                # Download directly to destination with resume support using deterministic dest_path
                self._download_from_url(
                    url=blob_url,
                    save_path=os.path.dirname(file_path),
                    progress_callback=_agg_progress,
                    filename=os.path.basename(file_path),
                    dest_path=file_path,
                )
                # Update aggregate counter: prefer reported blob size, fallback to actual file size
                try:
                    bytes_downloaded = start_offset + (int(blob.size or 0) or os.path.getsize(file_path))
                except Exception:
                    try:
                        bytes_downloaded = start_offset + os.path.getsize(file_path)
                    except Exception:
                        bytes_downloaded = start_offset
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
                # Common text/docs
                "md",
                "txt",
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
            import requests
            from concurrent.futures import ThreadPoolExecutor, as_completed
            from threading import Lock
            from huggingface_hub.utils import build_hf_headers
            from huggingface_hub import HfFolder, HfApi, hf_hub_url

            if hasattr(self, "logger"):
                self.logger.info(f"Downloading from Hugging Face Hub: {repo_id}")

            split_path = repo_id.split("/")

            if self._has_file_ending(repo_id):
                # fetch the specific file via signed URL and download through URL downloader
                self.logger.info(f"Downloading specific file from Hugging Face Hub: {repo_id}")
                base_repo = "/".join(split_path[:2])
                file_name = os.path.basename(repo_id)
                subfolder = f"{'/'.join(split_path[2:-1])}" if len(split_path) > 2 else None
                # deterministic final path based on logical repo path (not signed URL)
                deterministic_path = os.path.join(
                    save_path,
                    f"{hashlib.sha256(repo_id.encode()).hexdigest()}_{file_name}",
                )
                if os.path.exists(deterministic_path):
                    self.logger.info(f"File {deterministic_path} already exists, skipping download.")
                    return deterministic_path
                # Build resolve URL then follow redirects to signed URL
                resolve_url = hf_hub_url(repo_id=base_repo, filename=file_name, subfolder=subfolder)
                token = HfFolder.get_token()
                headers = build_hf_headers(token=token)
                with requests.Session() as sess:
                    sess.headers.update(headers)
                    head = sess.head(resolve_url, allow_redirects=True, timeout=20, verify=self._requests_verify())
                    head.raise_for_status()
                    signed_url = head.url
                    # Download directly to deterministic path with resume support
                    os.makedirs(os.path.dirname(deterministic_path), exist_ok=True)
                    self._download_from_url(
                        url=signed_url,
                        save_path=os.path.dirname(deterministic_path),
                        progress_callback=progress_callback,
                        session=sess,
                        filename=file_name,
                        dest_path=deterministic_path,
                    )
                self.logger.info(f"Successfully downloaded specific file from Hugging Face Hub: {repo_id}")
                return deterministic_path

            subfolder = (
                [f"{'/'.join(split_path[2:])}/*"] if len(split_path) > 2 else None
            )
            repo_id = "/".join(split_path if len(split_path) <= 2 else split_path[:2])

            dest_path = os.path.join(save_path, repo_id.replace("/", "_"))
            # If destination already exists and is non-empty:
            # - When downloading the whole repo (no subfolder), skip re-download.
            # - When downloading a subfolder, we will merge the staged subfolder into dest at finalize.
            if os.path.exists(dest_path) and os.listdir(dest_path) and not subfolder:
                self.logger.info(f"Directory {dest_path} already exists with content; skipping full repo re-download.")
                return dest_path
            # Snapshot-style: list files, resolve signed URLs for each, download in parallel, preserve structure
            api = HfApi()
            all_files = api.list_repo_files(repo_id=repo_id)
            # Restrict to subfolder if provided
            subfolder_root = None
            if subfolder:
                sub_path = subfolder[0].rstrip("*").rstrip("/")
                subfolder_root = sub_path
                # If the provided path matches an exact file in the repo, restrict to that file
                if sub_path in all_files:
                    all_files = [sub_path]
                else:
                    # Otherwise treat as a folder prefix
                    prefix = f"{sub_path}/" if sub_path else ""
                    all_files = [p for p in all_files if p.startswith(prefix)]
            if not all_files:
                self.logger.warning(f"No files found in repository: {repo_id}")
                return dest_path
            # Prepare session and headers
            token = HfFolder.get_token()
            headers = build_hf_headers(token=token)
            session = requests.Session()
            session.headers.update(headers)
            # First, resolve signed URLs and collect sizes to compute total
            file_entries = []
            total_size = 0
            for rel_path in all_files:
                # Split into subdir and filename
                filename = os.path.basename(rel_path)
                rel_dir = os.path.dirname(rel_path)
                resolve_url = hf_hub_url(repo_id=repo_id, filename=filename, subfolder=rel_dir or None)
                try:
                    head = session.head(resolve_url, allow_redirects=True, timeout=20, verify=self._requests_verify())
                    head.raise_for_status()
                    signed_url = head.url
                    size = None
                    try:
                        size = int(head.headers.get("content-length", "0")) or None
                    except Exception:
                        size = None
                    if size:
                        total_size += size
                    file_entries.append((rel_path, filename, rel_dir, signed_url, size))
                except Exception as e:
                    self.logger.error(f"Failed to resolve signed URL for {rel_path} in {repo_id}: {e}")
            if not file_entries:
                self.logger.warning(f"No downloadable files resolved in repository: {repo_id}")
                return dest_path
            # Progress aggregation across threads
            def make_cb(rel_path: str, size_hint: Optional[int]):
                def _cb(n: int, _total: Optional[int], _label: Optional[str] = None):
                    if progress_callback:
                        try:
                            # Report per-file progress with per-file total
                            progress_callback(int(n), _total if _total is not None else size_hint, _label or os.path.basename(rel_path))
                        except Exception:
                            pass
                return _cb
            # Download in parallel to temp (staging dir), then move the entire directory into dest_path atomically
            with tempfile.TemporaryDirectory() as tmp_root:
                stage_dir = os.path.join(tmp_root, os.path.basename(dest_path))
                os.makedirs(stage_dir, exist_ok=True)
                futures = []
                with ThreadPoolExecutor(max_workers=min(8, len(file_entries))) as ex:
                    for rel_path, filename, rel_dir, signed_url, _size in file_entries:
                        final_dir = os.path.join(stage_dir, rel_dir) if rel_dir else stage_dir
                        final_path = os.path.join(final_dir, filename)
                        # Skip if already present in staging (e.g., duplicate entries)
                        if os.path.exists(final_path):
                            if progress_callback:
                                try:
                                    size_inc = _size if _size is not None else os.path.getsize(final_path)
                                    if size_inc:
                                        progress_callback(size_inc, size_inc, filename)
                                except Exception:
                                    pass
                            continue
                        os.makedirs(final_dir, exist_ok=True)
                        # Each task downloads directly into staged final path with resume support
                        def task(rel_path=rel_path, filename=filename, signed_url=signed_url, rel_dir=rel_dir, final_dir=final_dir):
                            tmp_dir = os.path.join(tmp_root, "files_tmp", rel_dir) if rel_dir else os.path.join(tmp_root, "files_tmp")
                            os.makedirs(tmp_dir, exist_ok=True)
                            dest_file = os.path.join(final_dir, filename)
                            self._download_from_url(
                                url=signed_url,
                                save_path=tmp_dir,
                                progress_callback=make_cb(rel_path, _size),
                                session=session,
                                filename=filename,
                                dest_path=dest_file,
                            )
                            return dest_file
                        futures.append(ex.submit(task))
                    # ensure completion
                    for _ in as_completed(futures):
                        pass
                # After all downloads have completed successfully, move results into dest_path:
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                try:
                    if os.path.exists(dest_path):
                        # Merge staged content into existing destination without introducing empty dirs:
                        # - move only completed files (non-.part)
                        # - create parent dirs only when moving a file
                        # - skip overwriting existing files
                        files_to_move = []
                        for root, _dirs, files in os.walk(stage_dir):
                            for f in files:
                                if f.endswith(".part"):
                                    continue
                                src_f = os.path.join(root, f)
                                if not os.path.isfile(src_f):
                                    continue
                                files_to_move.append(src_f)
                        for src_f in files_to_move:
                            rel = os.path.relpath(src_f, stage_dir)
                            dst_f = os.path.join(dest_path, rel)
                            if os.path.exists(dst_f):
                                continue
                            os.makedirs(os.path.dirname(dst_f), exist_ok=True)
                            try:
                                os.replace(src_f, dst_f)
                            except Exception:
                                shutil.move(src_f, dst_f)
                        # Cleanup any empty directories left in staging
                        for root, dirs, files in os.walk(stage_dir, topdown=False):
                            for name in files:
                                # remove stray .part files from staging
                                if name.endswith(".part"):
                                    try:
                                        os.remove(os.path.join(root, name))
                                    except Exception:
                                        pass
                            try:
                                os.rmdir(root)
                            except Exception:
                                pass
                        self.logger.info(f"Merged staged Hugging Face files into existing destination: {dest_path}")
                    else:
                        try:
                            os.replace(stage_dir, dest_path)  # atomic on same filesystem when dest does not exist
                        except Exception:
                            shutil.move(stage_dir, dest_path)
                except Exception as e:
                    self.logger.error(f"Failed to finalize Hugging Face download into {dest_path}: {e}")
            if hasattr(self, "logger"):
                self.logger.info(f"Successfully downloaded from Hugging Face Hub: {repo_id}")
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
        fuzzy: bool = False,
    ):
        try:
            # Reuse gdown's robust Google Drive URL handling, but stream via our downloader
            # so we can report progress and share session/cookies.
            from gdown.download import _get_session, parse_url, get_url_from_gdrive_confirmation, FileURLRetrievalError  # type: ignore
            import re
            import textwrap
            import requests

            sess, cookies_file = _get_session(
                proxy=None,
                use_cookies=False,
                user_agent=DEFAULT_HEADERS["User-Agent"],
                return_cookies_file=True,
            )

            url_origin = url
            gdrive_file_id, is_gdrive_download_link = parse_url(url, warning=False)
            

            if fuzzy and gdrive_file_id:
                # Overwrite the url with fuzzy match of a file id
                url = "https://drive.google.com/uc?id={id}".format(id=gdrive_file_id)
                url_origin = url
                is_gdrive_download_link = True

            # Follow Google Drive confirmations/exports until we reach a direct file response
            while True:
                res = sess.get(url, stream=True, verify=self._requests_verify())
                if not (gdrive_file_id and is_gdrive_download_link):
                    break

                if url == url_origin and res.status_code == 500:
                    # The file could be Google Docs or Spreadsheets.
                    url = "https://drive.google.com/open?id={id}".format(id=gdrive_file_id)
                    continue

                if res.headers.get("Content-Type", "").startswith("text/html"):
                    m = re.search("<title>(.+)</title>", res.text)
                    if m and m.groups()[0].endswith(" - Google Docs"):
                        url = (
                            "https://docs.google.com/document/d/{id}/export"
                            "?format={format}".format(id=gdrive_file_id, format="docx")
                        )
                        continue
                    elif m and m.groups()[0].endswith(" - Google Sheets"):
                        url = (
                            "https://docs.google.com/spreadsheets/d/{id}/export"
                            "?format={format}".format(id=gdrive_file_id, format="xlsx")
                        )
                        continue
                    elif m and m.groups()[0].endswith(" - Google Slides"):
                        url = (
                            "https://docs.google.com/presentation/d/{id}/export"
                            "?format={format}".format(id=gdrive_file_id, format="pptx")
                        )
                        continue
                if "Content-Disposition" in res.headers:
                    # This is the file
                    break

                # Need to redirect with confirmation
                try:
                    url = get_url_from_gdrive_confirmation(res.text)
                    print(url)
                except FileURLRetrievalError as e:
                    message = (
                        "Failed to retrieve file url:\n\n{}\n\n"
                        "You may still be able to access the file from the browser:\n\n\t{}\n\n"
                        "but direct download failed. Please check connections and permissions."
                    ).format(
                        "\t" + "\n\t".join(textwrap.wrap(str(e))),
                        url_origin,
                    )
                    raise FileURLRetrievalError(message)

            # Determine filename from response headers if available
            def _filename_from_response(resp: requests.Response) -> Optional[str]:
                try:
                    cd = resp.headers.get("Content-Disposition") or resp.headers.get("content-disposition")
                    if not cd:
                        return None
                    # Basic parsing for filename="..."
                    match = re.search(r'filename\*?=(?:UTF-8\'\')?"?([^";]+)"?', cd)
                    if match:
                        return match.group(1)
                    return None
                except Exception:
                    return None

            filename_hint = _filename_from_response(res) or os.path.basename(urlparse(url).path) or "download"
            # Use our URL downloader with the prepared session and filename hint

            return self._download_from_url(
                url=url,
                save_path=save_path,
                progress_callback=progress_callback,
                session=sess,
                filename=filename_hint,
                stable_id=f"gdrive:{gdrive_file_id}" if gdrive_file_id else None,
            )
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
        session=None,
        filename: Optional[str] = None,
        chunk_size: int = 1024 * 1024, # 10MB
        adaptive: bool = True,
        initial_chunk_size: int = 512 * 1024,  # 512KB starting point for probing
        target_chunk_seconds: float = 0.25,    # aim ~250ms per read
        min_chunk_size: int = 64 * 1024,       # 64KB
        max_chunk_size: int = 16 * 1024 * 1024, # 16MB
        dest_path: Optional[str] = None,
        stable_id: Optional[str] = None,
    ):
        """Downloads a single file from a URL with resume support.
        
        If adaptive is True, dynamically adjusts chunk size during download based on observed
        throughput to target read durations around target_chunk_seconds, clamped to
        [min_chunk_size, max_chunk_size]."""
        import requests
        from tqdm import tqdm
        import time
        parsed_url = urlparse(url)
        relative_path_from_url = parsed_url.path.lstrip("/")
        # Compute deterministic destination
        if dest_path:
            file_path = dest_path
            base_name = os.path.basename(file_path)
        else:
            # Convert to hash of a stable key that ignores volatile query/fragment unless a stable_id is provided
            stable_key = stable_id if stable_id is not None else self._stable_url_key(url)
            file_name = hashlib.sha256(stable_key.encode('utf-8')).hexdigest()
            base_name = filename or os.path.basename(relative_path_from_url) or "download"
            file_path = os.path.join(save_path, f"{file_name}_{base_name}")
        part_path = f"{file_path}.part"
        # Name to use in logs/progress
        log_name = os.path.basename(file_path)
        try:
            # If the file already exists, do not re-download
            if os.path.exists(file_path):
                self.logger.info(f"File {file_path} already exists, skipping download.")
                return file_path

            # Prepare directory
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            # Determine resume offset if a partial file exists
            resume_size = 0
            if os.path.exists(part_path):
                try:
                    resume_size = os.path.getsize(part_path)
                except OSError:
                    resume_size = 0

            # Probe server for size/accept-ranges (best-effort)
            remote_size = None
            accept_ranges = False
            try:
                requester = session if session is not None else requests
                head_resp = requester.head(
                    url,
                    timeout=10,
                    verify=self._requests_verify(),
                    headers=DEFAULT_HEADERS,
                    allow_redirects=True,
                )
                if head_resp.ok:
                    try:
                        remote_size = int(head_resp.headers.get("content-length", "0")) or None
                    except Exception:
                        remote_size = None
                    accept_ranges = head_resp.headers.get("accept-ranges", "").lower() == "bytes"
            except Exception:
                # If HEAD fails, we'll still try GET (some servers don't support HEAD)
                pass

            # If partial already equals remote size, finalize without downloading
            if resume_size and remote_size and resume_size >= remote_size:
                try:
                    os.replace(part_path, file_path)
                    self.logger.info(f"Resumed and finalized existing partial file to {file_path}")
                    if progress_callback:
                        progress_callback(remote_size, remote_size, os.path.basename(file_path))
                    return file_path
                except Exception:
                    # If finalize fails, continue with resume flow
                    pass

            # Build headers; attempt Range resume if we have partial bytes
            headers = dict(DEFAULT_HEADERS)
            if resume_size > 0:
                headers["Range"] = f"bytes={resume_size}-"

            self.logger.info(
                f"Downloading {log_name} from: {url}{' (resuming at %d bytes)' % resume_size if resume_size else ''}"
            )

            get_requester = session if session is not None else requests
            with get_requester.get(
                url, timeout=10, verify=self._requests_verify(), headers=headers, stream=True, allow_redirects=True
            ) as response:
                # If we attempted to resume but server responded with 200, it ignored Range.
                # If we know total size and resume already matches it, finalize; otherwise restart from scratch.
                if resume_size > 0 and response.status_code == 200 and "Range" in headers:
                    if remote_size and resume_size >= remote_size:
                        try:
                            os.replace(part_path, file_path)
                            self.logger.info(f"Resumed and finalized existing partial file to {file_path}")
                            if progress_callback:
                                progress_callback(remote_size, remote_size, os.path.basename(file_path))
                            return file_path
                        except Exception:
                            pass
                    # Server does not support resume; restart download from scratch
                    self.logger.warning("Server did not honor Range header; restarting download from scratch.")
                    try:
                        if os.path.exists(part_path):
                            os.remove(part_path)
                    except Exception:
                        pass
                    resume_size = 0

                response.raise_for_status()

                # Determine total size for progress when possible
                total_size = remote_size
                content_range = response.headers.get("content-range")
                if content_range:
                    # Format: bytes start-end/total
                    try:
                        total_size = int(content_range.split("/")[-1])
                    except Exception:
                        pass
                if total_size is None:
                    try:
                        cl = int(response.headers.get("content-length", "0"))
                        total_size = cl + resume_size if cl and resume_size else (cl or None)
                    except Exception:
                        total_size = None
                
                # Ensure raw reads decode if needed (so sizes match actual written bytes)
                try:
                    response.raw.decode_content = True
                except Exception:
                    pass

                mode = "ab" if resume_size > 0 else "wb"
                downloaded_so_far = resume_size
                with open(part_path, mode) as out, tqdm(
                    desc=os.path.basename(file_path),
                    total=total_size,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    initial=resume_size,
                ) as bar:
                    if progress_callback and resume_size:
                        try:
                            progress_callback(downloaded_so_far, total_size or None, os.path.basename(file_path))
                        except Exception:
                            pass
                    
                    # Reading strategy: fixed-size via iter_content, or adaptive via response.raw.read
                    if not adaptive:
                        for chunk in response.iter_content(chunk_size=chunk_size):
                            if not chunk:
                                continue
                            out.write(chunk)
                            chunk_len = len(chunk)
                            bar.update(chunk_len)
                            downloaded_so_far += chunk_len
                            if progress_callback:
                                progress_callback(downloaded_so_far, total_size or None, os.path.basename(file_path))
                    else:
                        # Adaptive: start small, adjust size towards a target per-read duration
                        def clamp(n: int, low: int, high: int) -> int:
                            return low if n < low else (high if n > high else n)
                        current_chunk_size = clamp(int(initial_chunk_size), min_chunk_size, max_chunk_size)
                        speed_bps: Optional[float] = None
                        # If resuming and total is known, send an initial callback already done above
                        while True:
                            t0 = time.perf_counter()
                            try:
                                data = response.raw.read(current_chunk_size)
                            except Exception as _e:
                                # Fallback to iter_content on any raw read error
                                for chunk in response.iter_content(chunk_size=current_chunk_size):
                                    if not chunk:
                                        continue
                                    out.write(chunk)
                                    chunk_len = len(chunk)
                                    bar.update(chunk_len)
                                    downloaded_so_far += chunk_len
                                    if progress_callback:
                                        progress_callback(downloaded_so_far, total_size or None, os.path.basename(file_path))
                                break
                            if not data:
                                break
                            out.write(data)
                            chunk_len = len(data)
                            bar.update(chunk_len)
                            downloaded_so_far += chunk_len
                            if progress_callback:
                                progress_callback(downloaded_so_far, total_size or None, os.path.basename(file_path))
                            # Update speed estimate and adjust chunk size
                            dt = time.perf_counter() - t0
                            if dt > 0 and chunk_len > 0:
                                inst_bps = float(chunk_len) / dt
                                # Exponential moving average to smooth
                                if speed_bps is None:
                                    speed_bps = inst_bps
                                else:
                                    speed_bps = 0.7 * speed_bps + 0.3 * inst_bps
                                # Aim for reads around target_chunk_seconds
                                desired = int(speed_bps * max(0.05, target_chunk_seconds))
                                current_chunk_size = clamp(desired, min_chunk_size, max_chunk_size)

            # Finalize the downloaded file
            os.replace(part_path, file_path)
            self.logger.info(f"Successfully downloaded {log_name} to {file_path}")
        except Exception as e:
            self.logger.error(f"Failed to download from URL: {url}. Error: {e}")
        finally:
            return file_path

    def download_from_url(
        self,
        url: str,
        save_path: str,
        progress_callback: Optional[Callable[[int, Optional[int], Optional[str]], None]] = None,
        session=None,
        filename: Optional[str] = None,
        chunk_size: int = 1024 * 1024,
        adaptive: bool = True,
        initial_chunk_size: int = 512 * 1024,
        target_chunk_seconds: float = 0.25,
        min_chunk_size: int = 64 * 1024,
        max_chunk_size: int = 16 * 1024 * 1024,
        dest_path: Optional[str] = None,
        stable_id: Optional[str] = None,
    ):
        """Public wrapper to download a file from a URL with optional session and filename hint."""
        return self._download_from_url(
            url=url,
            save_path=save_path,
            progress_callback=progress_callback,
            session=session,
            filename=filename,
            chunk_size=chunk_size,
            adaptive=adaptive,
            initial_chunk_size=initial_chunk_size,
            target_chunk_seconds=target_chunk_seconds,
            min_chunk_size=min_chunk_size,
            max_chunk_size=max_chunk_size,
            dest_path=dest_path,
            stable_id=stable_id,
        )
