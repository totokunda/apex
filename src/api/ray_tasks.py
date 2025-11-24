"""
Ray tasks for preprocessor operations
"""
from typing import Dict, Any, Optional, Callable, List
from pathlib import Path
from urllib.parse import urlparse
import ray
import traceback
from loguru import logger
from src.preprocess.aux_cache import AuxillaryCache
import importlib
import os
import torch
import inspect
import yaml
from src.utils.cache import empty_cache
from src.api.preprocessor_registry import (
    get_preprocessor_info
)
from src.engine.registry import UniversalEngine 
from src.mixins.download_mixin import DownloadMixin
from src.utils.defaults import get_components_path, get_lora_path, get_preprocessor_path
from diffusers.utils import export_to_video
from src.lora.manager import LoraManager
from src.api.manifest import get_manifest, MANIFEST_BASE_PATH



def _derive_lora_name_from_source(source: str) -> str:
    """
    Derive a human-readable LoRA name/label from an arbitrary source string
    (local path, HF repo path, or URL).
    """
    try:
        src = (source or "").strip()
        parsed = urlparse(src)
        path = parsed.path if parsed.scheme else src
        path = path.rstrip("/") or src
        base = os.path.basename(path) or path
        # Drop query fragments if they somehow slipped through
        if "?" in base:
            base = base.split("?", 1)[0]
        if "#" in base:
            base = base.split("#", 1)[0]
        # Trim extension, keep stem only
        if "." in base:
            base = base.split(".", 1)[0]
        return base or src
    except Exception:
        return source



def _load_manifest_yaml(yaml_path: Path) -> Optional[Dict[str, Any]]:
    """Best-effort YAML loader that never raises; logs and returns None on failure."""
    try:
        if not yaml_path.exists():
            logger.warning(f"Manifest YAML not found at {yaml_path}")
            return None
        with yaml_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            logger.warning(f"Manifest YAML at {yaml_path} is not a mapping")
            return None
        return data
    except Exception as e:
        logger.error(f"Failed to load manifest YAML {yaml_path}: {e}")
        return None


def _save_manifest_yaml(yaml_path: Path, doc: Dict[str, Any]) -> None:
    """Best-effort YAML dumper; logs on failure and keeps existing file untouched."""
    try:
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        yaml_text = yaml.safe_dump(doc, sort_keys=False)
        yaml_path.write_text(yaml_text, encoding="utf-8")
    except Exception as e:
        logger.error(f"Failed to write updated manifest YAML {yaml_path}: {e}")


def _remove_lora_from_manifest(source: str, manifest_id: Optional[str] = None) -> bool:
    """
    Remove any LoRA entries whose source/remote_source (or raw string entry)
    matches `source` from the given manifest.

    Returns True if at least one entry was removed, otherwise False.
    """
    try:
        manifest = get_manifest(manifest_id)
        manifest_path = MANIFEST_BASE_PATH / Path(manifest.get("full_path"))
        doc = _load_manifest_yaml(manifest_path)
        if doc is None:
            return False

        spec = doc.get("spec") or {}
        loras = spec.get("loras") or []
        if not isinstance(loras, list):
            return False

        new_loras: List[Any] = []
        removed = False
        for entry in loras:
            if isinstance(entry, str):
                entry_source = entry
            elif isinstance(entry, dict):
                entry_source = entry.get("source") or entry.get("remote_source")
            else:
                entry_source = None

            if entry_source == source:
                removed = True
                continue
            new_loras.append(entry)

        if not removed:
            return False

        spec["loras"] = new_loras
        doc["spec"] = spec
        _save_manifest_yaml(manifest_path, doc)
        logger.info(f"Removed LoRA '{source}' from manifest {manifest_path.name}")
        return True
    except Exception as e:
        traceback.print_exc()
        logger.warning(f"Failed to remove LoRA '{source}' from manifest: {e}")
        return False


def _ensure_lora_registered_in_manifests(source: str, manifest_id: Optional[str] = None, lora_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Ensure that a LoRA entry for `source` exists in all engine manifests.

    Behaviour:
      - For each manifest YAML:
          - Ensures `spec.loras` exists as a list.
          - If an entry with matching `source` is absent, appends:
                - source: <source>
                - scale: 1.0
                - name: <derived name>
                - label: <derived name>
                - verified: false   (verification happens after download)
          - If entry already exists, leaves the rest of its fields untouched.
    """
    try:
        manifest = get_manifest(manifest_id)
        manifest_path = MANIFEST_BASE_PATH / Path(manifest.get("full_path"))
        doc = _load_manifest_yaml(manifest_path)
        if doc is None:
            return

        spec = doc.setdefault("spec", {})
        loras = spec.setdefault("loras", [])
        if not isinstance(loras, list):
            logger.warning(f"spec.loras is not a list in manifest {manifest_id}; leaving unchanged")
            return
        # Skip if already registered in this manifest
        already_present = False
        for entry in loras:
            if isinstance(entry, dict) and (entry.get("source") == source or entry.get("remote_source") == source):
                already_present = True
                break
            if isinstance(entry, str) and entry == source:
                already_present = True
                break
        if already_present:
            return entry
        name = lora_name or _derive_lora_name_from_source(source)
        new_entry: Dict[str, Any] = {
            "scale": 1.0,
            "name": name,
            "label": name,
            "verified": False,
        }
        loader = DownloadMixin()
        is_downloaded = loader.is_downloaded(source, get_lora_path())
        if is_downloaded:
            new_entry["source"] = is_downloaded
        else:
            new_entry["remote_source"] = source
        
        loras.append(new_entry)
        spec["loras"] = loras
        doc["spec"] = spec
        _save_manifest_yaml(manifest_path, doc)
        logger.info(f"Registered new LoRA '{source}' in manifest {doc.get('full_path')}")
        return new_entry
    except Exception as e: 
        traceback.print_exc()
        logger.warning(f"Failed to register LoRA '{source}' in manifests: {e}")
        return None


def _is_transformer_downloaded_for_manifest(doc: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Return the transformer component dict for this manifest with `model_path`
    set to a concrete local string path if any transformer weights are already
    downloaded.

    If nothing is downloaded or no transformer component is present, returns None.
    """
    try:
        spec = doc.get("spec") or {}
        components = spec.get("components") or []
        if not isinstance(components, list):
            return None

        for component in components:
            if not isinstance(component, dict):
                continue
            if component.get("type") != "transformer":
                continue

            raw_model_paths = component.get("model_path")
            candidate_paths: List[str] = []
            if isinstance(raw_model_paths, str):
                candidate_paths.append(raw_model_paths)
            elif isinstance(raw_model_paths, dict):
                path_val = raw_model_paths.get("path")
                if isinstance(path_val, str):
                    candidate_paths.append(path_val)
            elif isinstance(raw_model_paths, list):
                for item in raw_model_paths:
                    if isinstance(item, str):
                        candidate_paths.append(item)
                    elif isinstance(item, dict):
                        path_val = item.get("path")
                        if isinstance(path_val, str):
                            candidate_paths.append(path_val)

            base_dir = get_components_path()
            for p in candidate_paths:
                local = DownloadMixin.is_downloaded(str(p), base_dir)
                if local:
                    # Return a shallow copy so callers can safely mutate model_path
                    comp_copy: Dict[str, Any] = dict(component)
                    comp_copy["model_path"] = local
                    return comp_copy
        return None
    except Exception as e:
        logger.warning(f"Failed to determine transformer download status for manifest: {e}")
        return None


def _mark_lora_verified_in_manifests(source: str, manifest_id: Optional[str] = None) -> Optional[bool]:
    """
    Update manifest entries for `source` with a `verified` flag.

    - For each manifest that contains a matching LoRA `source` and whose transformer
      is already downloaded, set `verified: true` on that LoRA entry.

    Returns:
        - True  -> LoRA verified and kept
        - False -> LoRA determined invalid and removed from manifest
        - None  -> No changes made (e.g. transformer not available / no matching entries)
    """
    try:
        manifest = get_manifest(manifest_id)
        manifest_path = MANIFEST_BASE_PATH / Path(manifest.get("full_path"))
        doc = _load_manifest_yaml(manifest_path)
        if doc is None:
            return None
        spec = doc.get("spec") or {}
        loras = spec.get("loras") or []
        if not isinstance(loras, list):
            return None
        # Only attempt verification if the manifest's transformer is present locally
        transformer_component = _is_transformer_downloaded_for_manifest(doc)
        if not transformer_component:
            return None
        # Build engine once per manifest for validation
        try:
            engine = UniversalEngine(yaml_path=str(manifest_path), should_download=False).engine
        except Exception as e:
            logger.warning(f"Failed to create engine for manifest during LoRA validation: {e}")
            return None
        updated = False
        any_verified = False
        any_removed = False
        new_loras: List[Any] = []
        for entry in loras:
            # Determine this entry's source (if any)
            if isinstance(entry, str):
                entry_source = entry
            elif isinstance(entry, dict):
                entry_source = entry.get("source") or entry.get("remote_source")
            else:
                new_loras.append(entry)
                continue
            if entry_source != source:
                new_loras.append(entry)
                continue
            # Validate this LoRA against the transformer
            is_valid = False
            try:
                is_valid = bool(engine.validate_lora_path(entry_source, transformer_component))
            except Exception as ve:
                logger.warning(f"LoRA validation failed for '{entry_source}' in manifest {manifest_path}: {ve}")
                is_valid = False
            if not is_valid:
                # Drop invalid LoRAs from this manifest entirely
                logger.warning(f"Removing invalid LoRA '{entry_source}' from manifest {manifest_path}")
                updated = True
                any_removed = True
                # Do not append this entry to new_loras -> effectively remove it
                continue
            # Valid LoRA → normalize to dict with verified=True
            name = _derive_lora_name_from_source(entry_source)
            if isinstance(entry, str):
                entry = {
                    "source": entry_source,
                    "scale": 1.0,
                    "name": name,
                    "label": name,
                    "verified": True,
                }
            elif isinstance(entry, dict):
                entry = dict(entry)
                entry.setdefault("name", name)
                entry.setdefault("label", name)
                entry["verified"] = True
                entry["source"] = DownloadMixin.is_downloaded(entry_source, get_lora_path()) or entry.get("source")
            new_loras.append(entry)
            updated = True
        if not updated:
            return None
        spec["loras"] = new_loras
        doc["spec"] = spec
        _save_manifest_yaml(manifest_path, doc)
        if any_verified:
            logger.info(f"Updated LoRA '{source}' verification state in manifest {manifest_path.name}")
            return True
        if any_removed:
            logger.info(f"Removed invalid LoRA '{source}' from manifest {manifest_path.name}")
            return False
        return None
    except Exception as e:
        traceback.print_exc()
        logger.warning(f"Failed to update LoRA verification state for '{source}': {e}")
        return None

@ray.remote(num_cpus=0.1)
def download_unified(
    item_type: str,
    source: Any,
    job_id: str,
    ws_bridge,
    save_path: Optional[str] = None,
    manifest_id: Optional[str] = None,
    lora_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Unified downloader for components, LoRAs, and preprocessors.
    
    Behavior:
    - If item_type == "preprocessor" and source is a known preprocessor id, we initialize it via `.from_pretrained()`
      and mark it as downloaded, reporting progress over websocket.
    - Otherwise, we use DownloadMixin to download one or multiple paths directly into the appropriate default folder
      based on item_type (component, lora, preprocessor) or an explicit save_path override.
    
    Args:
        item_type: One of {"component", "lora", "preprocessor"}.
        source: A preprocessor id (string) OR a path/url/hf-repo (string) OR a list of such strings.
        job_id: Job ID for progress tracking.
        ws_bridge: Ray actor bridge to forward websocket updates.
        save_path: Optional override directory to save into; otherwise inferred from item_type.
        manifest_id: Optional manifest ID to use for LoRA verification.
        lora_name: Optional name to use for the LoRA.
    """
    # Helper to send progress
    logger.info(f"Downloading {item_type} {source} for job {job_id} with manifest {manifest_id} and lora name {lora_name}")
    def send_progress(progress: Optional[float], message: str, metadata: Optional[Dict[str, Any]] = None):
        try:
            ray.get(ws_bridge.send_update.remote(job_id, progress, message, metadata))
            if progress is not None:
                logger.info(f"[{job_id}] Progress: {progress*100:.1f}% - {message}")
            else:
                logger.info(f"[{job_id}] {message}")
        except Exception as e:
            logger.error(f"Failed to send progress update to websocket: {e}")
    
    try:
        norm_type = (item_type or "").strip().lower()
        if norm_type not in {"component", "lora", "preprocessor"}:
            raise ValueError(f"Unknown item_type '{item_type}'. Expected one of: component, lora, preprocessor.")
        
        # Determine default directory if not explicitly provided
        base_save_dir = save_path
        if base_save_dir is None:
            if norm_type == "component":
                base_save_dir = get_components_path()
            elif norm_type == "lora":
                base_save_dir = get_lora_path()
            else:
                base_save_dir = get_preprocessor_path()
        os.makedirs(base_save_dir, exist_ok=True)
        
        # Case 1: Preprocessor-id based download and initialization
        if norm_type == "preprocessor" and isinstance(source, str):
            try:
                preprocessor_info = get_preprocessor_info(source)

                # Force CPU in worker to avoid MPS/CUDA fork issues
                os.environ['CUDA_VISIBLE_DEVICES'] = ''
                os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
                if hasattr(torch, 'set_default_device'):
                    torch.set_default_device('cpu')
                
                send_progress(0.0, f"Starting download of preprocessor '{source}'")
                send_progress(0.1, "Loading preprocessor module")
                
                module = importlib.import_module(preprocessor_info["module"])
                preprocessor_class = getattr(module, preprocessor_info["class"])
                
                # Wire download progress into util
                from src.preprocess.download_tracker import DownloadProgressTracker
                from src.preprocess import util as util_module
                tracker = DownloadProgressTracker(job_id, lambda p, m, md=None: send_progress(p, m, md))
                util_module.DOWNLOAD_PROGRESS_CALLBACK = tracker.update_progress
                try:
                    preprocessor_class.from_pretrained()
                    from src.preprocess.base_preprocessor import BasePreprocessor
                    BasePreprocessor._mark_as_downloaded(source)
                finally:
                    util_module.DOWNLOAD_PROGRESS_CALLBACK = None
                
                send_progress(1.0, "Download complete")
                send_progress(1.0, "Complete", {"status": "complete"})
                return {
                    "job_id": job_id,
                    "status": "complete",
                    "type": "preprocessor",
                    "id": source,
                    "message": "Preprocessor downloaded and initialized",
                }
            except Exception as maybe_not_preproc:
                # Not a registered preprocessor id; fall through to generic downloader
                logger.info(f"'{source}' is not a registered preprocessor id or failed to init. Falling back to generic download. Reason: {maybe_not_preproc}")
                # continue to generic path-based downloading below
        
        elif norm_type == "lora" and isinstance(source, str):
            try:
                from src.preprocess.download_tracker import DownloadProgressTracker
                lora_manager = LoraManager()

                # 1) Immediately register the LoRA in manifests (if not already present)
                entry = _ensure_lora_registered_in_manifests(source, manifest_id, lora_name)
                
                logger.info(f"Registered LoRA '{source}' in manifests: {entry}")

                # 2) Start the actual download using the LoRA manager with progress tracking
                def _lora_download_progress_adapter(
                    p: Optional[float],
                    message: str,
                    metadata: Optional[Dict[str, Any]] = None,
                ):
                    """
                    Map raw download progress (0..1) into the 0..0.75 range of the
                    unified job progress, reserving the remaining 0.25 for LoRA
                    verification.
                    """
                    if p is None:
                        send_progress(None, message, metadata)
                        return
                    try:
                        frac = max(0.0, min(1.0, float(p)))
                    except Exception:
                        frac = 0.0
                    scaled = 0.75 * frac
                    send_progress(scaled, message, metadata)

                tracker = DownloadProgressTracker(job_id, _lora_download_progress_adapter)
                if entry is not None:
                    if "source" in entry and entry.get("verified"):
                        # No resolve needed
                        send_progress(
                            1.0,
                            "LoRA already downloaded and verified",
                            {
                                "status": "complete",
                                "bucket": norm_type,
                                "stage": "lora_download",
                            },
                        )
                        return {
                            "job_id": job_id,
                            "status": "complete",
                            "type": "lora",
                            "id": source,
                            "message": "LoRA already downloaded and verified",
                        }
                    else:
                        lora_item = lora_manager.resolve(
                            source,
                            prefer_name=lora_name,
                            progress_callback=tracker.update_progress,
                        )
                        if not getattr(lora_item, "local_paths", None) or len(lora_item.local_paths) == 0:
                            # Resolved but no actual files -> treat as failure and clean up manifest
                            logger.error(
                                f"No LoRA files were found for source '{source}'. "
                                "Removing LoRA entry from manifest."
                            )
                            _remove_lora_from_manifest(source, manifest_id)
                            send_progress(
                                0.0,
                                f"LoRA download failed for '{source}' (no files found)",
                                {
                                    "status": "error",
                                    "bucket": norm_type,
                                    "stage": "lora_download",
                                    "error": "No LoRA files found for source",
                                },
                            )
                            return {
                                "job_id": job_id,
                                "status": "error",
                                "type": "lora",
                                "id": source,
                                "message": "LoRA download failed: no files found for source; removed from manifest",
                            }
                else:
                    logger.error(f"LoRA entry is not registered in manifests: {entry}")
                    send_progress(0.0, f"LoRA entry is not registered in manifests: {entry}", {"status": "error", "bucket": norm_type, "stage": "lora_download", "error": f"LoRA entry is not registered in manifests: {entry}"})
                    return {
                        "job_id": job_id,
                        "status": "error",
                        "type": "lora",
                        "id": source,
                        "message": f"LoRA entry is not registered in manifests: {entry}",
                    }
                # Explicitly mark the end of the download phase at 75%
                try:
                    send_progress(
                        0.75,
                        "LoRA download complete",
                        {"status": "processing", "bucket": norm_type, "stage": "lora_download"},
                    )
                except Exception:
                    pass

                # 3) Verification phase: use the remaining 25% of progress
                try:
                    send_progress(
                        0.80,
                        "Verifying LoRA",
                        {"status": "processing", "bucket": norm_type, "stage": "lora_verification"},
                    )
                except Exception:
                    pass

                verified = _mark_lora_verified_in_manifests(lora_item.source, manifest_id)

                # If verification explicitly failed, surface an error and remove the LoRA
                if verified is False:
                    _remove_lora_from_manifest(lora_item.source, manifest_id)
                    send_progress(
                        0.0,
                        "LoRA verification failed; removed from manifest",
                        {
                            "status": "error",
                            "bucket": norm_type,
                            "stage": "lora_verification",
                            "verified": False,
                        },
                    )
                    return {
                        "job_id": job_id,
                        "status": "error",
                        "type": "lora",
                        "id": source,
                        "message": "LoRA verification failed; removed from manifest",
                    }

                try:
                    send_progress(
                        1.0,
                        "LoRA verification complete",
                        {
                            "status": "complete",
                            "bucket": norm_type,
                            "stage": "lora_verification",
                            "verified": bool(verified) if verified is not None else False,
                        },
                    )
                except Exception:
                    pass

                result: Dict[str, Any] = {
                    "job_id": job_id,
                    "status": "complete",
                    "type": "lora",
                    "id": source,
                    "message": "LoRA downloaded and initialized",
                    "local_paths": lora_item.local_paths,
                }
                if verified is not None:
                    result["verified"] = bool(verified)
                return result
            except Exception as maybe_not_lora:
                # Not a valid LoRA source or failed to download; clean up manifest and surface error
                logger.warning(
                    f"LoRA download/resolve failed for '{source}'. "
                    f"Removing LoRA from manifest. Reason: {maybe_not_lora}"
                )
                _remove_lora_from_manifest(source, manifest_id)
                send_progress(
                    0.0,
                    f"LoRA download failed for '{source}'",
                    {
                        "status": "error",
                        "bucket": norm_type,
                        "stage": "lora_download",
                        "error": str(maybe_not_lora),
                    },
                )
                return {
                    "job_id": job_id,
                    "status": "error",
                    "type": "lora",
                    "id": source,
                    "message": f"LoRA download failed; removed from manifest: {maybe_not_lora}",
                }
        
        # Case 2: Generic path/url/hf download(s) using DownloadMixin
        # Normalize sources to a list
        paths: List[str] = []
        if isinstance(source, list):
            paths = [str(p) for p in source]
        elif isinstance(source, str):
            paths = [source]
        else:
            raise ValueError("source must be a string or list of strings representing paths/urls/hf repos.")
        
        @ray.remote
        class ProgressAggregator:
            def __init__(self, total_items: int):
                self.total_items = max(1, int(total_items))
                # Per-item fractional progress (0..1) for fallback aggregation
                self.per_index_progress: Dict[int, float] = {}
                # Per-item byte accounting for more accurate aggregation on multi-path/folder downloads
                self.bytes_downloaded: Dict[int, int] = {}
                self.bytes_total: Dict[int, int] = {}
                self.last_overall: float = 0.0
            
            def update(
                self,
                index: int,
                frac: float,
                label: str,
                downloaded: Optional[int] = None,
                total: Optional[int] = None,
                filename: Optional[str] = None,
                message: Optional[str] = None,
            ):
                # Clamp per-item fraction
                frac = max(0.0, min(1.0, float(frac)))
                self.per_index_progress[index] = frac

                # Track byte-level progress when we know the total size
                if downloaded is not None and total is not None and total > 0:
                    try:
                        d = max(0, int(downloaded))
                        t = max(1, int(total))
                        # Ensure we never exceed the total for safety
                        self.bytes_downloaded[index] = min(d, t)
                        self.bytes_total[index] = t
                    except Exception:
                        # Best-effort only; fall back to fractional aggregation if anything goes wrong
                        pass

                # Prefer aggregate byte-based progress when we have enough information,
                # otherwise fall back to simple average of per-item fractions.
                overall_progress: float
                try:
                    total_bytes = sum(self.bytes_total.values())
                    if total_bytes > 0:
                        done_bytes = 0
                        for idx, t in self.bytes_total.items():
                            d = self.bytes_downloaded.get(idx, 0)
                            if d < 0:
                                d = 0
                            if d > t:
                                d = t
                            done_bytes += d
                        overall_progress = max(0.0, min(1.0, done_bytes / float(total_bytes)))
                    else:
                        # Fallback: average of known item fractions
                        overall_progress = sum(self.per_index_progress.values()) / float(self.total_items)
                except Exception:
                    overall_progress = sum(self.per_index_progress.values()) / float(self.total_items)

                # Clamp to [0, 1]; allow progress to move down slightly when a new file
                # in a multi-file/folder download starts (so we reflect aggregate progress
                # across all bytes, not just the current file).
                overall_progress = max(0.0, min(1.0, overall_progress))
                self.last_overall = overall_progress

                if filename:
                    filename_parts = filename.split("_")
                    if len(filename_parts) > 1:
                        filename_parts = filename_parts[1:]
                    filename = "_".join(filename_parts)

                meta = {"label": label, "bucket": norm_type}
                if downloaded is not None:
                    meta["downloaded"] = int(downloaded)
                if total is not None:
                    meta["total"] = int(total)
                if filename is not None:
                    meta["filename"] = filename
                msg = message or f"Downloading {label}"
                try:
                    # Report aggregated overall progress to the websocket, not per-file fraction
                    return ray.get(ws_bridge.send_update.remote(job_id, overall_progress, msg, meta))
                except Exception:
                    return False
            
            def complete(self, index: int, label: str):
                return self.update(index, 1.0, label, message=f"Completed {label}")
            
            def error(self, index: int, label: str, error_msg: str):
                try:
                    return ray.get(ws_bridge.send_update.remote(job_id, self.last_overall, error_msg, {"label": label, "status": "error", "bucket": norm_type}))
                except Exception:
                    return False
        
        @ray.remote
        def download_single(path: str, dest_dir: str, index: int, aggregator) -> Dict[str, Any]:
            label = os.path.basename(path.rstrip("/")) or path
            try:
                def _cb(downloaded: int, total: Optional[int], filename: Optional[str] = None):
                    frac = 0.0
                    if total and total > 0:
                        frac = max(0.0, min(1.0, downloaded / total))
                    ray.get(aggregator.update.remote(index, frac, label, downloaded, total, filename))
                mixin = DownloadMixin()
                os.makedirs(dest_dir, exist_ok=True)
                mixin.logger.info(f"[{job_id}] Downloading {path} into {dest_dir}")
                result_path = mixin.download(path, dest_dir, progress_callback=_cb)
                ray.get(aggregator.complete.remote(index, label))
                return {"path": path, "status": "complete", "result_path": result_path}
            except Exception as e:
                ray.get(aggregator.error.remote(index, label, str(e)))
                return {"path": path, "status": "error", "error": str(e)}
        
        total = len(paths)
        aggregator = ProgressAggregator.remote(total)
        refs = [download_single.remote(p, base_save_dir, i, aggregator) for i, p in enumerate(paths, start=1)]
        results = ray.get(refs)
        try:
            ray.get(ws_bridge.send_update.remote(job_id, 1.0, "All downloads complete", {"status": "complete", "bucket": norm_type}))
        except Exception:
            pass
        has_error = any(r.get("status") == "error" for r in results)
        return {
            "job_id": job_id,
            "status": "error" if has_error else "complete",
            "bucket": norm_type,
            "save_dir": base_save_dir,
            "results": results,
        }
    except Exception as e:
        tb = traceback.format_exc()
        logger.error(tb)
        try:
            send_progress(0.0, str(e), {"status": "error", "error": str(e)})
        except Exception:
            pass
        return {"job_id": job_id, "status": "error", "error": str(e), "traceback": tb}


def _execute_preprocessor(
    preprocessor_name: str,
    input_path: str,
    job_id: str,
    send_progress: Callable[[Optional[float], str, Optional[Dict[str, Any]]], None],
    start_frame: Optional[int] = None,
    end_frame: Optional[int] = None,
    **kwargs,
) -> Dict[str, Any]:
    preprocessor_info = get_preprocessor_info(preprocessor_name)
    cache = AuxillaryCache(
        input_path,
        preprocessor_name,
        start_frame,
        end_frame,
        kwargs,
        supports_alpha_channel=preprocessor_info.get("supports_alpha_channel", False),
    )
    
    media_type = cache.type
    send_progress(0.05, "Checking cache")

    if cache.is_cached():
        send_progress(1.0, "Cache found and returning")
        send_progress(1.0, "Complete", {"status": "complete"})
        return {
            "job_id": job_id,
            "status": "complete",
            "result_path": cache.get_result_path(),
            "type": media_type,
        }

    send_progress(0.1, "Loading preprocessor module")
    module = importlib.import_module(preprocessor_info["module"])
    preprocessor_class = getattr(module, preprocessor_info["class"])

    from src.preprocess.download_tracker import DownloadProgressTracker
    from src.preprocess import util as util_module

    tracker = DownloadProgressTracker(
        job_id,
        lambda p, m, md=None: send_progress(
            0.05 + (max(0.0, min(1.0, float(p))) * 0.15), m, md
        ),
    )
    util_module.DOWNLOAD_PROGRESS_CALLBACK = tracker.update_progress
    try:
        preprocessor = preprocessor_class.from_pretrained()
    finally:
        util_module.DOWNLOAD_PROGRESS_CALLBACK = None

    send_progress(0.2, "Preprocessor loaded")

    from src.preprocess.base_preprocessor import BasePreprocessor

    BasePreprocessor._mark_as_downloaded(preprocessor_name)

    def progress_callback(idx: int, total: int, message: str = None):
        total = max(1, int(total))
        frac = idx / float(total)
        scaled_progress = 0.2 + (max(0.0, min(1.0, frac)) * 0.6)
        send_progress(scaled_progress, message or f"Processing frame {idx} of {total}")

    try:
        if media_type == "video":
            frame_range = cache._get_video_frame_range()
            total_frames = len([f for f in frame_range if f not in cache.cached_frames])
            frames = cache.video_frames(batch_size=1)
            result = preprocessor(
                frames,
                job_id=job_id,
                progress_callback=progress_callback,
                total_frames=total_frames,
                **kwargs,
            )
        else:
            result = preprocessor(cache.image, job_id=job_id, **kwargs)

        result_path = cache.save_result(result)
        send_progress(1.0, "Result saved")
        send_progress(1.0, "Complete", {"status": "complete"})

        return {
            "status": "complete",
            "result_path": result_path,
            "type": cache.type,
        }

    except Exception as e:
        error_msg = f"Error processing {preprocessor_name}: {str(e)}"
        error_traceback = traceback.format_exc()
        logger.error(f"[{job_id}] Processing failed: {error_traceback}")
        try:
            send_progress(0.0, error_msg, {"status": "error", "error": error_msg})
        except Exception as ws_error:
            logger.error(
                f"[{job_id}] Processing failed AND websocket notification failed: {error_msg}, WS Error: {ws_error}"
            )
        return {
            "job_id": job_id,
            "status": "error",
            "error": error_msg,
            "traceback": error_traceback,
        }

    finally:
        empty_cache()


@ray.remote
def run_preprocessor(
    preprocessor_name: str,
    input_path: str,
    job_id: str,
    ws_bridge,
    start_frame: Optional[int] = None,
    end_frame: Optional[int] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Run a preprocessor on input media
    """

    def send_progress(progress: float, message: str, metadata: Optional[Dict] = None):
        """Local send_progress that uses the passed ws_bridge"""
        try:
            ray.get(ws_bridge.send_update.remote(job_id, progress, message, metadata))
            logger.info(f"[{job_id}] Progress: {progress*100:.1f}% - {message}")
        except Exception as e:
            logger.error(f"Failed to send progress update to websocket: {e}")

    return _execute_preprocessor(
        preprocessor_name,
        input_path,
        job_id,
        send_progress,
        start_frame=start_frame,
        end_frame=end_frame,
        **kwargs,
    )


@ray.remote
def run_engine_from_manifest(
    manifest_path: str,
    job_id: str,
    ws_bridge,
    inputs: Dict[str, Any],
    selected_components: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Execute a manifest YAML with provided inputs and persist result to disk."""
    def send_progress(progress: float | None, message: str, metadata: Optional[Dict] = None):
        try:
            ray.get(ws_bridge.send_update.remote(job_id, progress, message, metadata))
            if progress is not None:
                logger.info(f"[{job_id}] Progress: {progress*100:.1f}% - {message}")
            else:
                logger.info(f"[{job_id}] Progress: {message}")
        except Exception as e:
            logger.error(f"Failed to send progress update: {e}")

    # Track large objects so we can explicitly drop references in a finally block
    engine = None
    raw = None
    config = None
    prepared_inputs: Dict[str, Any] = {}
    preprocessor_jobs: List[Dict[str, Any]] = []

    try:
        from src.utils.yaml import load_yaml as load_manifest_yaml
        from src.manifest.loader import validate_and_normalize
        from src.engine.registry import UniversalEngine
        from src.utils.defaults import DEFAULT_CACHE_PATH
        import numpy as np
        from PIL import Image
        
        logger.info(manifest_path, "manifest_path")

        # Normalize manifest (handles v1 -> engine shape)
        raw = load_manifest_yaml(manifest_path)
        config = validate_and_normalize(raw)
        inputs = inputs or {}
        selected_components = selected_components or {}
        
        def _extract_ui_inputs() -> List[Dict[str, Any]]:
            spec_ui = ((raw.get("spec") or {}).get("ui") or {})
            raw_inputs = spec_ui.get("inputs")
            if isinstance(raw_inputs, list):
                return raw_inputs
            normalized_ui = (config.get("ui") or {}).get("inputs")
            if isinstance(normalized_ui, list):
                return normalized_ui
            return []

        preprocessor_map: Dict[str, str] = {}
        for item in _extract_ui_inputs():
            if not isinstance(item, dict):
                continue
            input_id = item.get("id")
            preproc_ref = item.get("preprocessor_ref")
            if input_id and preproc_ref:
                preprocessor_map[input_id] = preproc_ref
        
        # Resolve engine settings
        engine_type = config.get("engine") or (config.get("spec") or {}).get("engine")
        model_type = config.get("type") or (config.get("spec") or {}).get("model_type")
        if isinstance(model_type, list):
            model_type = model_type[0] if model_type else None
            
        attention_type = selected_components.pop("attention", {}).get("name", None)
        
        input_kwargs = {
            "engine_type": engine_type,
            "yaml_path": manifest_path,
            "model_type": model_type,
            "selected_components": selected_components,
            **(config.get("engine_kwargs", {}) or {}),
        }

        
        if attention_type:
            input_kwargs["attention_type"] = attention_type
            
        
        engine = UniversalEngine(**input_kwargs)


        def _coerce_media_input(value: Any) -> tuple[Optional[str], Optional[bool]]:
            if isinstance(value, dict):
                path_candidate = (
                    value.get("input_path")
                    or value.get("src")
                    or value.get("path")
                )
                apply_flag = value.get("apply_preprocessor")
                path_str = path_candidate if isinstance(path_candidate, str) else None
                apply_bool = apply_flag if isinstance(apply_flag, bool) else None
                return path_str, apply_bool
            if isinstance(value, str):
                return value, None
            return None, None

        prepared_inputs: Dict[str, Any] = {}
        preprocessor_jobs: List[Dict[str, Any]] = []
        for input_key, raw_value in inputs.items():
            if input_key in preprocessor_map: 
                media_path, apply_flag = _coerce_media_input(raw_value)
                if media_path:
                    prepared_inputs[input_key] = media_path
                    should_apply = apply_flag if isinstance(apply_flag, bool) else True
                    preprocessor_kwargs = raw_value.get("preprocessor_kwargs", {})
                    if should_apply:
                        preprocessor_jobs.append(
                            {
                                "input_id": input_key,
                                "preprocessor_name": preprocessor_map[input_key],
                                "input_path": media_path,
                                "preprocessor_kwargs": preprocessor_kwargs,
                            }
                        )
                else:
                    prepared_inputs[input_key] = raw_value
            else:
                prepared_inputs[input_key] = raw_value

        # Prepare job directory early (needed for previews)
        job_dir = Path(DEFAULT_CACHE_PATH) / "engine_results" / (job_id)
        job_dir.mkdir(parents=True, exist_ok=True)

        # Unified saver usable for previews and final outputs
        def save_output(output_obj, filename_prefix: str = "result", final: bool = False):
            result_path: Optional[str] = None
            media_type: Optional[str] = None
            try:
                # String path passthrough
                if isinstance(output_obj, str):
                    result_path = output_obj
                    media_type = "path"
                # Single image
                elif isinstance(output_obj, Image.Image):
                    ext = f"png" if final else "jpg"
                    result_path = str(job_dir / f"{filename_prefix}.{ext}")
                    output_obj.save(result_path)
                    media_type = "image"
                # Sequence of frames
                elif isinstance(output_obj, list) and len(output_obj) > 0:
                    fps = (
                        config.get("spec", {}).get("fps")
                        or (config or {}).get("defaults", {}).get("run", {}).get("fps")
                    )
                    if not fps:
                        try:
                            impl = getattr(engine.engine, "implementation_engine", None)
                            if impl is not None:
                                sig = inspect.signature(impl.run)
                                param = sig.parameters.get("fps")
                                if param is not None and param.default is not inspect._empty:
                                    fps = param.default
                        except Exception:
                            pass
                    fps = fps or 16
                    result_path = str(job_dir / f"{filename_prefix}.mp4")
                    export_to_video(output_obj, result_path, fps=int(fps), quality=8.0 if final else 5.0)
                    media_type = "video"
                
                else:
                    # Fallback best-effort serialization
                    try:
                        arr = np.asarray(output_obj)  # type: ignore[arg-type]
                        result_path = str(job_dir / f"{filename_prefix}.png")
                        Image.fromarray(arr).save(result_path)
                        media_type = "image"
                    except Exception as e:
                        logger.error(f"Failed to save output: {e}")
                        result_path = str(job_dir / f"{filename_prefix}.txt")
                        with open(result_path, "w") as f:
                            f.write(str(type(output_obj)))
                        media_type = "unknown"
            except Exception as save_err:
                traceback.print_exc()
                logger.error(f"Failed to save output: {save_err}")
                raise
            return result_path, media_type

        total_steps = max(1, len(preprocessor_jobs) + 1)
        
        logger.info(f"Total steps: {total_steps}")
        logger.info(f"Preprocessor jobs: {preprocessor_jobs}")

        for idx, job in enumerate(preprocessor_jobs):
            stage_start = idx / total_steps
            stage_span = 1.0 / total_steps

            def stage_send_progress(local_progress: Optional[float], message: str, metadata: Optional[Dict] = None):
                merged_meta = dict(metadata or {})
                merged_meta.setdefault("stage", "preprocessor")
                merged_meta.setdefault("input_id", job["input_id"])
                if merged_meta.get("status") == "complete":
                    merged_meta["status"] = "processing"
                if local_progress is None:
                    send_progress(None, message, merged_meta)
                    return
                bounded = max(0.0, min(1.0, float(local_progress)))
                send_progress(stage_start + bounded * stage_span, message, merged_meta)

            stage_send_progress(
                0.0,
                f"Running {job['preprocessor_name']} preprocessor for {job['input_id']}",
            )
            result = _execute_preprocessor(
                job["preprocessor_name"],
                job["input_path"],
                f"{job_id}:{job['input_id']}",
                stage_send_progress,
                **job.get("preprocessor_kwargs", {}),
            )
            if result.get("status") != "complete":
                raise RuntimeError(
                    result.get("error")
                    or f"Preprocessor {job['preprocessor_name']} failed"
                )
            print(f"\n\n\n {result} \n\n\n")
            prepared_inputs[job["input_id"]] = result.get("result_path")

        engine_stage_start = len(preprocessor_jobs) / total_steps
        engine_stage_span = 1.0 / total_steps

        # Render-on-step callback that writes previews
        step_counter = {"i": 0}
        def render_on_step_callback(frames):
            try:
                idx = step_counter["i"]
                step_counter["i"] = idx + 1
                # Persist preview to cache and notify over websocket with metadata only
                result_path, media_type = save_output(frames, filename_prefix=f"preview_{idx:04d}")
                logger.info(f"Preview saved to {result_path} with media type {media_type}")
                try:
                    # Send an update that does not overwrite progress (progress=None)
                    logger.info(f"Sending preview websocket update at step {idx} with result path {result_path} and media type {media_type}")
                    send_progress(None, f"Preview frame {idx}", {
                        "status": "preview",
                        "preview_path": result_path,
                        "type": media_type,
                        "index": idx,
                    })
                except Exception as se:
                    logger.warning(f"Failed sending preview websocket update at step {idx}: {se}")
            except Exception as e:
                logger.warning(f"Preview save failed at step {step_counter['i']}: {e}")

        # Progress callback forwarded into the engine
        def progress_callback(progress: float, message: str, metadata: Optional[Dict] = None):
            logger.info(f"Progress callback: {progress}, {message}, {metadata}")
            if progress is None:
                send_progress(None, message, metadata)
                return
            bounded = max(0.0, min(1.0, progress))
            send_progress(engine_stage_start + bounded * engine_stage_span, message, metadata)
        
        import json 
        json.dump({
                "engine_kwargs": input_kwargs,
                "inputs": prepared_inputs,
            }, indent=4, fp=open("inputs.json", "w"))
        
        output = engine.run(
            **(prepared_inputs or {}),
            progress_callback=progress_callback,
            render_on_step=True,
            render_on_step_callback=render_on_step_callback,
        )
        
        # Persist final result using the unified saver
        result_path, media_type = save_output(output[0], filename_prefix="result", final=True)

        send_progress(1.0, "Complete", {"status": "complete"})
        return {"status": "complete", "result_path": result_path, "type": media_type}

    except Exception as e:
        tb = traceback.format_exc()
        logger.error(tb, "traceback")
        try:
            send_progress(0.0, str(e), {"status": "error", "error": str(e)})
        except Exception:
            pass
        return {"job_id": job_id, "status": "error", "error": str(e), "traceback": tb}
    finally:
        # Ensure we aggressively release references and clear CUDA/MPS caches
        try:
            engine = None
            raw = None
            config = None
            prepared_inputs = {}
            preprocessor_jobs = []
        except Exception as cleanup_err:
            logger.warning(f"run_engine_from_manifest cleanup failed: {cleanup_err}")
        empty_cache()


@ray.remote
def run_frame_interpolation(
    input_path: str,
    target_fps: float,
    job_id: str,
    ws_bridge,
    exp: Optional[int] = None,
    scale: float = 1.0,
) -> Dict[str, Any]:
    """Run RIFE frame interpolation on a video and save an output video.

    Args:
        input_path: Path to input video file
        target_fps: Desired output frames per second
        job_id: Job id for websocket/job tracking
        ws_bridge: Ray actor bridge for websocket updates
        exp: Optional exponent for 2**exp interpolation (overrides target_fps if provided)
        scale: RIFE scale (for UHD set 0.5)

    Returns:
        Dict with status, result_path and type
    """
    def send_update(progress: float | None, message: str, metadata: Optional[Dict[str, Any]] = None):
        try:
            ray.get(ws_bridge.send_update.remote(job_id, progress, message, metadata))
        except Exception:
            pass

    from pathlib import Path
    from src.utils.defaults import DEFAULT_CACHE_PATH
    try:
        from src.postprocess.rife.rife import RifePostprocessor
        send_update(0.05, "Initializing RIFE")
        pp = RifePostprocessor(target_fps=target_fps, exp=exp, scale=scale)

        send_update(0.15, "Running frame interpolation")

        # Wire progress from postprocessor (scale 0.2 -> 0.95)
        def frame_progress(idx: int, total: int, message: Optional[str] = None):
            try:
                total = max(1, int(total))
                frac = max(0.0, min(1.0, float(idx) / float(total)))
                scaled = 0.20 + frac * 0.75
                send_update(scaled, message or f"Interpolating {idx}/{total}")
            except Exception:
                pass

        frames = pp(
            input_path,
            target_fps=target_fps,
            exp=exp,
            scale=scale,
            progress_callback=frame_progress,
        )
        
        # Save output video (video-only first), then mux original audio if present
        import subprocess
        import shutil

        job_dir = Path(DEFAULT_CACHE_PATH) / "postprocessor_results" / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        video_only_path = str(job_dir / "result_video.mp4")
        final_out_path = str(job_dir / "result.mp4")

        fps_to_write = int(max(1, round(target_fps)))
        export_to_video(frames, video_only_path, fps=fps_to_write, quality=8.0)

        # Try to mux audio from input_path into the final output without changing rate/tempo
        # If no audio is present, fall back to the video-only file
        try:
            # Use ffmpeg with stream copy to preserve original audio rate/tempo
            # -map 0:v:0 takes video from the first input (our generated video)
            # -map 1:a:0? takes the first audio track from the second input if it exists
            ffmpeg_cmd = [
                "ffmpeg",
                "-y",
                "-i", video_only_path,
                "-i", input_path,
                "-map", "0:v:0",
                "-map", "1:a:0?",
                "-c:v", "copy",
                "-c:a", "copy",
                "-shortest",
                "-movflags", "+faststart",
                final_out_path,
            ]
            proc = subprocess.run(ffmpeg_cmd, capture_output=True)
            if proc.returncode != 0:
                # If muxing failed (e.g., no audio stream), just use the video-only output
                shutil.move(video_only_path, final_out_path)
        except Exception as e:
            logger.error(f"Failed to mux audio: {e}")
            # On any unexpected error, fall back to video-only output
            try:
                shutil.move(video_only_path, final_out_path)
            except Exception:
                # If move also fails, keep path consistent
                final_out_path = video_only_path

        send_update(1.0, "Complete", {"status": "complete", "result_path": final_out_path, "type": "video"})
        return {"status": "complete", "result_path": final_out_path, "type": "video"}
    except Exception as e:
        tb = traceback.format_exc()
        logger.error(tb)
        try:
            send_update(0.0, str(e), {"status": "error", "error": str(e)})
        except Exception:
            pass
        return {"job_id": job_id, "status": "error", "error": str(e), "traceback": tb}
