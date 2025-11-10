"""
Ray tasks for preprocessor operations
"""
from typing import Dict, Any, Optional, Callable, List
from pathlib import Path
import ray
import traceback
from loguru import logger
from src.preprocess.aux_cache import AuxillaryCache
import importlib
import os
import torch
import inspect
from src.utils.cache import empty_cache
from src.api.preprocessor_registry import (
    get_preprocessor_info
)
from src.mixins.download_mixin import DownloadMixin
from src.utils.defaults import get_components_path
from diffusers.utils import export_to_video
import time


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
        from src.preprocess.download_tracker import DownloadProgressTracker
        tracker = DownloadProgressTracker(job_id, lambda p, m, md=None: send_progress(p, m, md))
        
        # Monkey patch the download tracker into the module
        from src.preprocess import util as util_module
        util_module.DOWNLOAD_PROGRESS_CALLBACK = tracker.update_progress
        
        # Download/initialize
        logger.info(f"Downloading model files for {preprocessor_name}")
        
        try:
            preprocessor = preprocessor_class.from_pretrained()
            logger.info(f"Preprocessor loaded successfully: {preprocessor_name}")
            
            # Mark as downloaded in tracking file
            from src.preprocess.base_preprocessor import BasePreprocessor
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
                    if should_apply:
                        preprocessor_jobs.append(
                            {
                                "input_id": input_key,
                                "preprocessor_name": preprocessor_map[input_key],
                                "input_path": media_path,
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
            )
            if result.get("status") != "complete":
                raise RuntimeError(
                    result.get("error")
                    or f"Preprocessor {job['preprocessor_name']} failed"
                )
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
