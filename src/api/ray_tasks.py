"""
Ray tasks for preprocessor operations
"""
import asyncio
from typing import Dict, Any, Optional
import ray
from src.api.ws_manager import websocket_manager
import traceback
from loguru import logger
from src.auxillary.aux_cache import AuxillaryCache
import importlib
import os
import torch
from src.utils.cache import empty_cache
from src.api.preprocessor_registry import (
    get_preprocessor_info,
    list_preprocessors,
    get_preprocessor_details
)
from src.mixins.download_mixin import DownloadMixin
from src.utils.defaults import get_components_path
from typing import List


@ray.remote
def download_preprocessor(preprocessor_name: str, job_id: str, ws_bridge) -> Dict[str, Any]:
    """
    Download and initialize a preprocessor model
    
    Args:
        preprocessor_name: Name of the preprocessor
        job_id: Job ID for tracking
        
    Returns:
        Dictionary with download status
    """
    def send_progress(progress: float, message: str, metadata: Optional[Dict] = None):
        """Local send_progress that uses the passed ws_bridge"""
        try:
            ray.get(ws_bridge.send_update.remote(job_id, progress, message, metadata))
            logger.info(f"[{job_id}] Progress: {progress*100:.1f}% - {message}")
        except Exception as e:
            logger.error(f"Failed to send progress update to websocket: {e}")
    
    try:
        # Force CPU usage in workers to avoid MPS/CUDA fork issues
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
        
        if hasattr(torch, 'set_default_device'):
            torch.set_default_device('cpu')
        
        logger.info(f"Starting download of {preprocessor_name}")
        send_progress(0.0, f"Starting download of {preprocessor_name}")
        
        # Get preprocessor info
        preprocessor_info = get_preprocessor_info(preprocessor_name)
        
        # Import module
        logger.info(f"Loading preprocessor module for {preprocessor_name}")
        send_progress(0.1, "Loading preprocessor module")
        module = importlib.import_module(preprocessor_info["module"])
        preprocessor_class = getattr(module, preprocessor_info["class"])
        
        # Setup download progress tracking
        from src.auxillary.download_tracker import DownloadProgressTracker
        tracker = DownloadProgressTracker(job_id, lambda p, m, md=None: send_progress(p, m, md))
        
        # Monkey patch the download tracker into the module
        from src.auxillary import util as util_module
        util_module.DOWNLOAD_PROGRESS_CALLBACK = tracker.update_progress
        
        # Download/initialize
        logger.info(f"Downloading model files for {preprocessor_name}")
        
        try:
            preprocessor = preprocessor_class.from_pretrained()
            logger.info(f"Preprocessor loaded successfully: {preprocessor_name}")
            
            # Mark as downloaded in tracking file
            from src.auxillary.base_preprocessor import BasePreprocessor
            BasePreprocessor._mark_as_downloaded(preprocessor_name)
            
        except Exception as load_error:
            logger.error(f"Failed to load preprocessor {preprocessor_name}: {str(load_error)}")
            raise
        finally:
            # Always clear the callback
            util_module.DOWNLOAD_PROGRESS_CALLBACK = None
        
        logger.info(f"Download complete for {preprocessor_name}")
        send_progress(1.0, "Download complete")
        
        # Send final completion status
        send_progress(1.0, "Complete", {"status": "complete"})
        
        return {
            "job_id": job_id,
            "status": "complete",
            "preprocessor": preprocessor_name,
            "message": "Download completed successfully"
        }
        
    except Exception as e:
        error_msg = f"Error downloading {preprocessor_name}: {str(e)}"
        error_traceback = traceback.format_exc()
        
        # Try to send error status to websocket
        try:
            send_progress(0.0, error_msg, {"status": "error", "error": error_msg})
            logger.error(f"[{job_id}] Download failed: {error_msg}")
        except Exception as ws_error:
            logger.error(f"[{job_id}] Download failed AND websocket notification failed: {error_msg}, WS Error: {ws_error}")
        
        return {
            "job_id": job_id,
            "status": "error",
            "error": error_msg,
            "traceback": error_traceback
        }


@ray.remote
def download_components(paths: List[str], job_id: str, ws_bridge, save_path: Optional[str] = None) -> Dict[str, Any]:
    """Download a list of component paths concurrently with aggregated websocket progress."""
    @ray.remote
    class ComponentsProgressAggregator:
        def __init__(self, job: str, bridge, total_items: int):
            self.job_id = job
            self.bridge = bridge
            self.total_items = max(1, total_items)
            self.per_index_progress: Dict[int, float] = {}
            self.last_overall: float = 0.0

        def update(self, index: int, frac: float, label: str, downloaded: Optional[int] = None, total: Optional[int] = None, filename: Optional[str] = None, message: Optional[str] = None):
            frac = max(0.0, min(1.0, float(frac)))
            self.per_index_progress[index] = frac
            total_progress = sum(self.per_index_progress.values()) / float(self.total_items)
            overall = max(self.last_overall, min(1.0, total_progress))
            self.last_overall = overall
            meta = {"label": label}
            if downloaded is not None:
                meta["downloaded"] = int(downloaded)
            if total is not None:
                meta["total"] = int(total)
            if filename is not None:
                meta["filename"] = filename
            if message:
                msg = message
            else:
                msg = f"Downloading {label}"
            try:
                return ray.get(self.bridge.send_update.remote(self.job_id, overall, msg, meta))
            except Exception:
                return False

        def complete(self, index: int, label: str):
            return self.update(index, 1.0, label, message=f"Completed {label}")

        def error(self, index: int, label: str, error_msg: str):
            try:
                return ray.get(self.bridge.send_update.remote(self.job_id, self.last_overall, error_msg, {"label": label, "status": "error"}))
            except Exception:
                return False

    @ray.remote
    def download_component_single(path: str, save_dir: str, index: int, total_items: int, aggregator) -> Dict[str, Any]:
        label = os.path.basename(path.rstrip("/")) or path
        try:
            def _cb(downloaded: int, total: Optional[int], filename: Optional[str] = None):
                frac = 0.0
                if total and total > 0:
                    frac = max(0.0, min(1.0, downloaded/total))
                
                ray.get(aggregator.update.remote(index, frac, label, downloaded, total, filename))
            mixin = DownloadMixin()
            mixin.logger.info(f"Downloading component {path} to {save_dir}")
            mixin.download(path, save_dir, progress_callback=_cb)
            ray.get(aggregator.complete.remote(index, label))
            return {"path": path, "status": "complete"}
        except Exception as e:
            ray.get(aggregator.error.remote(index, label, str(e)))
            return {"path": path, "status": "error", "error": str(e)}

    try:
        save_dir = save_path or get_components_path()
        os.makedirs(save_dir, exist_ok=True)

        total_items = len(paths)
        aggregator = ComponentsProgressAggregator.remote(job_id, ws_bridge, total_items)
        # fire off all downloads in parallel
        refs = []
        for idx, path in enumerate(paths, start=1):
            refs.append(download_component_single.remote(path, save_dir, idx, total_items, aggregator))

        results = ray.get(refs)
        # mark completion
        try:
            ray.get(ws_bridge.send_update.remote(job_id, 1.0, "All downloads complete", {"status": "complete"}))
        except Exception:
            pass
        # determine overall status
        has_error = any(r.get("status") == "error" for r in results)
        return {"job_id": job_id, "status": "error" if has_error else "complete", "results": results}
    except Exception as e:
        err = str(e)
        try:
            ray.get(ws_bridge.send_update.remote(job_id, 0.0, err, {"status": "error", "error": err}))
        except Exception:
            pass
        return {"job_id": job_id, "status": "error", "error": err}


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
    
    Args:
        preprocessor_name: Name of the preprocessor
        input_path: Path to input image or video
        job_id: Job ID for tracking
        start_frame: Start frame index for video (None = from beginning)
        end_frame: End frame index for video (None = to end)
        **kwargs: Additional preprocessing parameters
        
    Returns:
        Dictionary with processing results
    """
    def send_progress(progress: float, message: str, metadata: Optional[Dict] = None):
        """Local send_progress that uses the passed ws_bridge"""
        try:
            ray.get(ws_bridge.send_update.remote(job_id, progress, message, metadata))
            logger.info(f"[{job_id}] Progress: {progress*100:.1f}% - {message}")
        except Exception as e:
            logger.error(f"Failed to send progress update to websocket: {e}")
            
        
    
    preprocessor_info = get_preprocessor_info(preprocessor_name)
    cache = AuxillaryCache(input_path, preprocessor_name, start_frame, end_frame, kwargs, supports_alpha_channel=preprocessor_info.get("supports_alpha_channel", False))
    media_type = cache.type
    send_progress(0.05, "Checking cache")
    
    # TODO: re-enable cache for testing purposes
    if cache.is_cached():
        send_progress(1.0, "Cache found and returning")
        send_progress(1.0, "Complete", {"status": "complete"})
        return {
            "job_id": job_id,
            "status": "complete",
            "result_path": cache.get_result_path(),
            "type": media_type,
        }
    
    
    # Progressive download/init: scale progress from 0.05 -> 0.20
    send_progress(0.1, "Loading preprocessor module")
    module = importlib.import_module(preprocessor_info["module"])
    preprocessor_class = getattr(module, preprocessor_info["class"])

    # Setup download progress tracking similar to download_preprocessor but scaled to 20%
    from src.auxillary.download_tracker import DownloadProgressTracker
    from src.auxillary import util as util_module
    tracker = DownloadProgressTracker(job_id, lambda p, m, md=None: send_progress(0.05 + (max(0.0, min(1.0, float(p))) * 0.15), m, md))
    util_module.DOWNLOAD_PROGRESS_CALLBACK = tracker.update_progress
    try:
        preprocessor = preprocessor_class.from_pretrained()
    finally:
        # Always clear the callback
        util_module.DOWNLOAD_PROGRESS_CALLBACK = None
    send_progress(0.2, "Preprocessor loaded")
    
    # Mark as downloaded in tracking file (in case it was loaded for the first time here)
    from src.auxillary.base_preprocessor import BasePreprocessor
    BasePreprocessor._mark_as_downloaded(preprocessor_name)
    
    def progress_callback(idx: int, total: int, message: str = None):
        progress = idx / total
        scaled_progress = 0.2 + (progress * 0.6)
        send_progress(scaled_progress, message or f"Processing frame {idx} of {total}")
    
    try:
        if media_type == "video":
            # Get frame generator and total count for progress tracking
            frame_range = cache._get_video_frame_range()
            print(f"Frame range: {frame_range}", start_frame, end_frame)
            total_frames = len([f for f in frame_range if f not in cache.cached_frames])
            frames = cache.video_frames(batch_size=1)
            result = preprocessor(frames, job_id=job_id, progress_callback=progress_callback, total_frames=total_frames, **kwargs)
        else:
            result = preprocessor(cache.image, job_id=job_id, **kwargs)
               
        result_path = cache.save_result(result)
        send_progress(1.0, "Result saved")
        
        # Send final completion status
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
        
        # Try to send error status to websocket - ensure client knows job failed
        try:
            send_progress(0.0, error_msg, {"status": "error", "error": error_msg})
        except Exception as ws_error:
            logger.error(f"[{job_id}] Processing failed AND websocket notification failed: {error_msg}, WS Error: {ws_error}")
        
        return {
            "job_id": job_id,
            "status": "error",
            "error": error_msg,
            "traceback": error_traceback
        }
    
    finally:
        empty_cache()

