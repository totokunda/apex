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
    PREPROCESSOR_REGISTRY,
    get_preprocessor_info,
    list_preprocessors,
    get_preprocessor_details
)

# Legacy registry reference (imported from preprocessor_registry)
_LEGACY_REGISTRY = {
    "anime_face_segment": {
        "module": "src.auxillary.anime_face_segment",
        "class": "AnimeFaceSegmentor"
    },
    "binary": {
        "module": "src.auxillary.binary",
        "class": "BinaryDetector",
    },
    "canny": {
        "module": "src.auxillary.canny",
        "class": "CannyDetector",
    },
    "color": {
        "module": "src.auxillary.color",
        "class": "ColorDetector",
    },
    "densepose": {
        "module": "src.auxillary.densepose",
        "class": "DenseposeDetector",
    },
    "depth_anything": {
        "module": "src.auxillary.depth_anything.transformers",
        "class": "DepthAnythingDetector",
    },
    "depth_anything_v2": {
        "module": "src.auxillary.depth_anything_v2",
        "class": "DepthAnythingV2Detector",
    },
    "diffusion_edge": {
        "module": "src.auxillary.diffusion_edge",
        "class": "DiffusionEdgeDetector",
    },
    "dsine": {
        "module": "src.auxillary.dsine",
        "class": "DsineDetector",
    },
    "dwpose": {
        "module": "src.auxillary.dwpose",
        "class": "DwposeDetector",
    },
    "animalpose": {
        "module": "src.auxillary.dwpose",
        "class": "AnimalPoseDetector",
    },
    "hed": {
        "module": "src.auxillary.hed",
        "class": "HEDdetector",
    },
    "leres": {
        "module": "src.auxillary.leres",
        "class": "LeresDetector",
    },
    "lineart": {
        "module": "src.auxillary.lineart",
        "class": "LineartDetector",
    },
    "lineart_anime": {
        "module": "src.auxillary.lineart_anime",
        "class": "LineartAnimeDetector",
    },
    "lineart_standard": {
        "module": "src.auxillary.lineart_standard",
        "class": "LineartStandardDetector",
    },
    "manga_line": {
        "module": "src.auxillary.manga_line",
        "class": "LineartMangaDetector",
    },
    "mediapipe_face": {
        "module": "src.auxillary.mediapipe_face",
        "class": "MediapipeFaceDetector",
    },
    "mesh_graphormer": {
        "module": "src.auxillary.mesh_graphormer",
        "class": "MeshGraphormerDetector",
    },
    "metric3d": {
        "module": "src.auxillary.metric3d",
        "class": "Metric3DDetector",
    },
    "midas": {
        "module": "src.auxillary.midas.transformers",
        "class": "MidasDetector",
    },
    "mlsd": {
        "module": "src.auxillary.mlsd",
        "class": "MLSDdetector",
    },
    "normalbae": {
        "module": "src.auxillary.normalbae",
        "class": "NormalBaeDetector",
    },
    "oneformer": {
        "module": "src.auxillary.oneformer.transformers",
        "class": "OneformerSegmentor",
    },
    "open_pose": {
        "module": "src.auxillary.open_pose",
        "class": "OpenposeDetector",
    },
    "pidi": {
        "module": "src.auxillary.pidi",
        "class": "PidiNetDetector",
    },
    "ptlflow": {
        "module": "src.auxillary.ptlflow",
        "class": "PTLFlowDetector",
    },
    "pyracanny": {
        "module": "src.auxillary.pyracanny",
        "class": "PyraCannyDetector",
    },
    "rembg": {
        "module": "src.auxillary.rembg",
        "class": "RembgDetector",
    },
    "recolor": {
        "module": "src.auxillary.recolor",
        "class": "Recolorizer",
    },
    "scribble": {
        "module": "src.auxillary.scribble",
        "class": "ScribbleDetector",
    },
    "scribble_xdog": {
        "module": "src.auxillary.scribble",
        "class": "ScribbleXDogDetector",
    },
    "scribble_anime": {
        "module": "src.auxillary.scribble_anime",
        "class": "ScribbleAnimeDetector",
    },
    "shuffle": {
        "module": "src.auxillary.shuffle",
        "class": "ContentShuffleDetector",
    },
    "teed": {
        "module": "src.auxillary.teed",
        "class": "TEDDetector",
    },
    "tile": {
        "module": "src.auxillary.tile",
        "class": "TileDetector",
    },
    "tile_gf": {
        "module": "src.auxillary.tile",
        "class": "TTPlanet_Tile_Detector_GF",
    },
    "tile_simple": {
        "module": "src.auxillary.tile",
        "class": "TTPLanet_Tile_Detector_Simple",
    },
    "uniformer": {
        "module": "src.auxillary.uniformer",
        "class": "UniformerSegmentor",
    },
    "unimatch": {
        "module": "src.auxillary.unimatch",
        "class": "UnimatchDetector",
    },
    "zoe": {
        "module": "src.auxillary.zoe.transformers",
        "class": "ZoeDetector",
    },
    "zoe_depth_anything": {
        "module": "src.auxillary.zoe.transformers",
        "class": "ZoeDepthAnythingDetector",
    },
    
}

# Note: get_preprocessor_info, list_preprocessors, and get_preprocessor_details 
# are now imported from preprocessor_registry module


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
        send_progress(0.2, "Downloading model files")
        
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
    
    cache = AuxillaryCache(input_path, preprocessor_name, start_frame, end_frame, kwargs)
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
    
    preprocessor_info = get_preprocessor_info(preprocessor_name)
    preprocessor_class = getattr(importlib.import_module(preprocessor_info["module"]), preprocessor_info["class"])
    preprocessor = preprocessor_class.from_pretrained()
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
            frames = cache.video
            result = preprocessor(frames, job_id=job_id, progress_callback=progress_callback, **kwargs)
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

