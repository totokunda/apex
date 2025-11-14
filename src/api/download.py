from __future__ import annotations
from typing import Any, Dict, List, Optional, Union, Tuple
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, validator
import uuid
import json
import os
import shutil
import ray
from loguru import logger
from pathlib import Path

from .ws_manager import get_ray_ws_bridge
from .job_store import register_job, job_store as unified_job_store
from .ray_tasks import download_unified
from src.utils.defaults import get_components_path, get_lora_path, get_preprocessor_path
from src.mixins.download_mixin import DownloadMixin
from .preprocessor_registry import check_preprocessor_downloaded


router = APIRouter(prefix="/download", tags=["download"])

# In-memory cache mapping request keys -> job_id
_request_key_to_job_id: Dict[str, str] = {}

# Fixed namespace for deterministic job ids
_NAMESPACE_UUID = uuid.UUID("8b0b145f-8f8b-4a7b-9c5a-9a6ca3e7a001")


def _normalize_item_type(item_type: str) -> str:
    t = (item_type or "").strip().lower()
    if t not in {"component", "lora", "preprocessor"}:
        raise ValueError("item_type must be one of: component, lora, preprocessor")
    return t


def _default_save_dir_for(item_type: str) -> str:
    if item_type == "component":
        return get_components_path()
    if item_type == "lora":
        return get_lora_path()
    return get_preprocessor_path()


def _canonical_source(source: Union[str, List[str]]) -> Union[str, List[str]]:
    if isinstance(source, list):
        # Canonicalize list order/content for deterministic keys
        return sorted([str(s).strip() for s in source])
    return str(source).strip()


def _request_key(item_type: str, source: Union[str, List[str]], save_path: Optional[str]) -> str:
    canonical = {
        "item_type": _normalize_item_type(item_type),
        "source": _canonical_source(source),
        "save_path": os.path.abspath(save_path) if save_path else None,
    }
    return json.dumps(canonical, sort_keys=True, separators=(",", ":"))


def _deterministic_job_id(item_type: str, source: Union[str, List[str]], save_path: Optional[str]) -> str:
    key = _request_key(item_type, source, save_path)
    return str(uuid.uuid5(_NAMESPACE_UUID, key))


def _already_downloaded(item_type: str, source: Union[str, List[str]], save_path: Optional[str]) -> Tuple[bool, str]:
    """
    Returns (downloaded, base_dir)
    """
    itype = _normalize_item_type(item_type)
    base_dir = save_path or _default_save_dir_for(itype)
    os.makedirs(base_dir, exist_ok=True)

    # Preprocessor id check
    if itype == "preprocessor" and isinstance(source, str):
        try:
            if check_preprocessor_downloaded(source):
                return True, base_dir
        except Exception:
            pass
        # Fallthrough to generic path checks in case source was not an id but a path/url

    # Generic path/url/hf repo check
    sources: List[str] = [source] if isinstance(source, str) else list(source)
    all_present = True
    for s in sources:
        local = DownloadMixin.is_downloaded(str(s), base_dir)
        if not local:
            all_present = False
            break
    return all_present, base_dir


class UnifiedDownloadRequest(BaseModel):
    item_type: str = Field(..., description="One of: component, lora, preprocessor")
    source: Union[str, List[str]]
    save_path: Optional[str] = None
    job_id: Optional[str] = None

    @validator("item_type")
    def _valid_item_type(cls, v: str) -> str:
        return _normalize_item_type(v)


class JobResponse(BaseModel):
    job_id: str
    status: str
    message: Optional[str] = None


class ResolveRequest(BaseModel):
    item_type: str
    source: Union[str, List[str]]
    save_path: Optional[str] = None

    @validator("item_type")
    def _valid_item_type(cls, v: str) -> str:
        return _normalize_item_type(v)


class ResolveResponse(BaseModel):
    job_id: str
    exists: bool
    running: bool
    downloaded: bool
    bucket: str
    save_dir: str
    source: Union[str, List[str]]


class BatchResolveRequest(BaseModel):
    item_type: str
    sources: List[Union[str, List[str]]]
    save_path: Optional[str] = None

    @validator("item_type")
    def _valid_item_type(cls, v: str) -> str:
        return _normalize_item_type(v)


class BatchResolveResponse(BaseModel):
    results: List[ResolveResponse]


class DeleteRequest(BaseModel):
    path: str
    item_type: Optional[str] = None  # Optional: 'component' | 'lora' | 'preprocessor'
    source: Optional[Union[str, List[str]]] = None  # Optional: to unmark preprocessor and clear mappings
    save_path: Optional[str] = None  # Optional: used to target mapping cleanup

    @validator("item_type")
    def _valid_item_type(cls, v: Optional[str]) -> Optional[str]:
        return _normalize_item_type(v) if v else v


class DeleteResponse(BaseModel):
    path: str
    status: str
    removed_mapping: bool = False
    unmarked: bool = False


@router.post("", response_model=JobResponse)
def start_unified_download(request: UnifiedDownloadRequest):
    """
    Start a unified download job. Progress can be observed via:
    - WebSocket: /ws/job/{job_id}
    - Polling:   /download/status/{job_id} (or /jobs/status/{job_id})
    Cancel via:   /download/cancel/{job_id} (or /jobs/cancel/{job_id})
    """
    try:
        # Determine deterministic job id and request key
        job_id = request.job_id or _deterministic_job_id(request.item_type, request.source, request.save_path)
        req_key = _request_key(request.item_type, request.source, request.save_path)
        # If a job with this request key was already started, return existing job id
        existing_job_id = _request_key_to_job_id.get(req_key)
        if existing_job_id:
            status_info = unified_job_store.status(existing_job_id)
            status = status_info.get("status", "unknown")
            if status in {"running", "queued"}:
                return JobResponse(job_id=existing_job_id, status=status, message="Existing job in progress")
            # If completed and assets are present, reuse id
            downloaded, _ = _already_downloaded(request.item_type, request.source, request.save_path)
            if downloaded:
                return JobResponse(job_id=existing_job_id, status="complete", message="Already downloaded")
            # Otherwise start a new one below

        # If already downloaded and no need to start a job
        downloaded, _ = _already_downloaded(request.item_type, request.source, request.save_path)
        if downloaded:
            # Cache the deterministic id mapping even when not starting a Ray job
            _request_key_to_job_id[req_key] = job_id
            return JobResponse(job_id=job_id, status="complete", message="Already downloaded")

        # Start the Ray job
        # Clear any cached/stale websocket state for this deterministic job_id to avoid replaying "complete"
        try:
            from .ws_manager import websocket_manager
            websocket_manager.clear_latest(job_id)
        except Exception:
            pass
        bridge = get_ray_ws_bridge()
        try:
            # Also clear any pending Ray-actor buffered updates for this job_id
            ray.get(bridge.clear_updates.remote(job_id))
        except Exception:
            pass
        ref = download_unified.remote(
            request.item_type,
            request.source,
            job_id,
            bridge,
            request.save_path,
        )
        register_job(job_id, ref, "download", {
            "item_type": request.item_type,
            "source": request.source,
            "save_path": request.save_path,
            "request_key": req_key,
        })
        _request_key_to_job_id[req_key] = job_id
        logger.info(f"Started unified download job {job_id} for {request.item_type}")
        return JobResponse(job_id=job_id, status="queued", message="Download job created")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start unified download: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/resolve", response_model=ResolveResponse)
def resolve_job_id(request: ResolveRequest):
    """
    Resolve or compute the job_id for a given request (item_type + source [+ save_path]).
    Returns whether a matching job exists, is running, and whether the assets are already downloaded.
    """
    try:
        req_key = _request_key(request.item_type, request.source, request.save_path)
        job_id = _request_key_to_job_id.get(req_key) or _deterministic_job_id(request.item_type, request.source, request.save_path)
        downloaded, base_dir = _already_downloaded(request.item_type, request.source, request.save_path)

        running = False
        exists = False
        if job_id:
            info = unified_job_store.get(job_id)
            exists = info is not None
            if exists:
                status = unified_job_store.status(job_id).get("status", "unknown")
                running = status in {"running", "queued"}

        return ResolveResponse(
            job_id=job_id,
            exists=exists,
            running=running,
            downloaded=downloaded,
            bucket=_normalize_item_type(request.item_type),
            save_dir=base_dir,
            source=request.source,
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to resolve job id: {e}")


@router.get("/status/{job_id}")
def download_status(job_id: str) -> Dict[str, Any]:
    """Polling status for a unified download job."""
    return unified_job_store.status(job_id)


@router.post("/cancel/{job_id}", response_model=JobResponse)
def cancel_download(job_id: str) -> JobResponse:
    """Cancel a unified download job."""
    result = unified_job_store.cancel(job_id)
    status = result.get("status", "unknown")
    message = result.get("message")
    if status in ["cancelled", "canceled"]:
        return JobResponse(job_id=job_id, status=status, message=message)
    raise HTTPException(status_code=404, detail=message or "Job not found")


@router.post("/resolve/batch", response_model=BatchResolveResponse)
def resolve_job_ids_batch(request: BatchResolveRequest):
    """
    Resolve job_ids for multiple sources at once. Each result mirrors /download/resolve.
    """
    try:
        results: List[ResolveResponse] = []
        for src in request.sources or []:
            req_key = _request_key(request.item_type, src, request.save_path)
            job_id = _request_key_to_job_id.get(req_key) or _deterministic_job_id(request.item_type, src, request.save_path)
            downloaded, base_dir = _already_downloaded(request.item_type, src, request.save_path)

            running = False
            exists = False
            if job_id:
                info = unified_job_store.get(job_id)
                exists = info is not None
                if exists:
                    status = unified_job_store.status(job_id).get("status", "unknown")
                    running = status in {"running", "queued"}

            results.append(ResolveResponse(
                job_id=job_id,
                exists=exists,
                running=running,
                downloaded=downloaded,
                bucket=_normalize_item_type(request.item_type),
                save_dir=base_dir,
                source=src,
            ))
        return BatchResolveResponse(results=results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to resolve batch job ids: {e}")


@router.delete("/delete", response_model=DeleteResponse)
def delete_downloaded_path(request: DeleteRequest):
    """
    Delete a downloaded file or directory from the filesystem.
    Safety checks ensure deletion is within known download roots unless explicitly scoped by item_type.
    Also clears deterministic job-id mapping for the corresponding request and unmarks preprocessor downloads.
    """
    try:
        # Determine allowed base(s)
        allowed_bases: List[Path] = []
        if request.item_type:
            base = request.save_path or _default_save_dir_for(request.item_type)
            allowed_bases.append(Path(base).resolve())
        else:
            # Default to all known buckets if not specified
            allowed_bases.extend([
                Path(get_components_path()).resolve(),
                Path(get_lora_path()).resolve(),
                Path(get_preprocessor_path()).resolve(),
            ])

        target = Path(request.path)
        target_resolved = (allowed_bases[0] / target).resolve() if not target.is_absolute() else target.resolve()

        # Ensure within one of the allowed roots
        def _is_within_any_base(p: Path, bases: List[Path]) -> bool:
            for b in bases:
                try:
                    p.relative_to(b)
                    return True
                except ValueError:
                    continue
            return False

        if not _is_within_any_base(target_resolved, allowed_bases):
            raise HTTPException(status_code=400, detail="path must be within an allowed download directory")

        # Perform deletion
        if not target_resolved.exists():
            raise HTTPException(status_code=404, detail="Path not found")
        if target_resolved.is_dir():
            shutil.rmtree(target_resolved, ignore_errors=False)
        else:
            target_resolved.unlink()

        # Best-effort: prune empty parent directories up to (but not including) the allowed base
        try:
            containing_base: Optional[Path] = None
            for b in allowed_bases:
                try:
                    target_resolved.relative_to(b)
                    containing_base = b
                    break
                except ValueError:
                    continue
            if containing_base:
                current = target_resolved.parent
                while current != containing_base and current.exists() and current.is_dir():
                    try:
                        # Only remove if empty; stop when a non-empty directory is encountered
                        if any(current.iterdir()):
                            break
                        next_parent = current.parent
                        current.rmdir()
                        current = next_parent
                    except Exception:
                        # If we cannot remove a directory for any reason, stop cleanup
                        break
        except Exception:
            pass

        removed_mapping = False
        unmarked = False

        # Best-effort: clear deterministic request -> job_id mappings that match item_type+source (any save_path)
        try:
            if request.item_type and request.source is not None:
                norm_type = _normalize_item_type(request.item_type)
                norm_source = _canonical_source(request.source)
                to_delete: List[str] = []
                for key, jid in _request_key_to_job_id.items():
                    try:
                        parsed = json.loads(key)
                        if parsed.get("item_type") == norm_type and parsed.get("source") == norm_source:
                            to_delete.append(key)
                    except Exception:
                        continue
                for k in to_delete:
                    del _request_key_to_job_id[k]
                    removed_mapping = True
        except Exception:
            pass

        # If this was a preprocessor id deletion, unmark it
        try:
            if request.item_type == "preprocessor" and isinstance(request.source, str) and request.source:
                from src.preprocess.base_preprocessor import BasePreprocessor
                BasePreprocessor._unmark_as_downloaded(request.source)
                unmarked = True
        except Exception:
            pass

        return DeleteResponse(
            path=str(target_resolved),
            status="deleted",
            removed_mapping=removed_mapping,
            unmarked=unmarked,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete path: {e}")

