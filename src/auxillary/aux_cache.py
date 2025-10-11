"""
Intelligent frame caching system for video preprocessing
"""
import hashlib
import json
from pathlib import Path
from typing import Dict,  Optional
import decord
import numpy as np
from src.utils.defaults import DEFAULT_CACHE_PATH
import os
from src.types import OutputMedia
from src.mixins.loader_mixin import LoaderMixin
from PIL import Image
import imageio

class AuxillaryCache:
    """
    Manages cached frames for video processing to avoid reprocessing
    """
    
    def __init__(self, path: str, preprocessor_name: str, start_frame: Optional[int] = None, end_frame: Optional[int] = None, params: Dict = None):
        """
        Initialize frame cache
        
        Args:
            path: Path to input video
            preprocessor_name: Name of preprocessor
            params: Processing parameters (used for cache key)
        """
        self.path = path
        self._media_type = None
        self.preprocessor_name = preprocessor_name
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.params = params
        self.cache_key = self._create_cache_key()
        self.cache_path = Path(DEFAULT_CACHE_PATH) / self.cache_key
        self.metadata_path = self.cache_path / "metadata.json"
        self.result_path = self.cache_path / "result.png" if self.type == "image" else self.cache_path / "result.mp4"
        # make sure the cache path exists
        self._video = None
        self._video_info = None
        self._image = None
        self._metadata = None
        self.cached_frames = set()
        self.non_cached_frames = dict()
        
        
        os.makedirs(self.cache_path, exist_ok=True)
        if self.type == "video":
            self._video_info, self._video = self._get_video()
            self._metadata = self.load_metadata()
            self.cached_frames = set(self._metadata["cached_frames"]) if self._metadata else set()
        else:
            self._image = self._get_image()

    def _create_cache_key(self):
        """
        Create a deterministic cache key from all relevant inputs.
        Uses MD5 for speed (sufficient for cache keys) and includes all parameters.
        """
        # Get the base filename without extension for readability
        file_stem = Path(self.path).stem[:32]  # Limit length
        
        # Create a canonical representation of all cache-relevant data
        cache_data = {
            "path": self.path,
            "preprocessor": self.preprocessor_name,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "params": self.params or {}
        }
        
        # Use sort_keys to ensure consistent JSON serialization regardless of dict ordering
        cache_json = json.dumps(cache_data, sort_keys=True)
        
        # MD5 is faster than SHA256 and sufficient for cache keys (not security-critical)
        cache_hash = hashlib.md5(cache_json.encode()).hexdigest()[:16]  # 16 chars is enough
        
        # Combine readable prefix with hash for easier debugging
        return f"{file_stem}_{self.preprocessor_name}_{cache_hash}"
    
    def load_metadata(self):
        if self.metadata_path.exists():
            with open(self.metadata_path, 'r') as f:
                return json.load(f)
        else:
            return None
    
    def _check_cache(self):
        if self.type == "image":
            # check if the result path exists
            if self.result_path.exists():
                return True
            else:
                return False
        ## we need to check if the metadata path exists and which frames are cached. 
        if self._metadata is not None:
            cached_frames = self.cached_frames
            # check if all frames in frame_range are cached
            return all(frame in cached_frames for frame in self._get_video_frame_range())
        else:
            return False
        
    def _get_image(self):
        return Image.open(self.path)  
    
    def _get_video_frame_range(self):
        start_frame = self.start_frame if self.start_frame is not None else 0
        end_frame = self.end_frame if self.end_frame is not None else self._video_info["frame_count"]
        return range(start_frame, end_frame)
    
    def _get_video(self):
        try:
            video = decord.VideoReader(self.path)
        except Exception as e:
            raise ValueError(f"Cannot open video file: {self.path}") from e
        return {
            "fps": video.get_avg_fps(),
            "width": video[0].shape[1],
            "height": video[0].shape[0],
            "frame_count": len(video),
        }, video
        
    def is_cached(self):
        return self._check_cache()

    def get_result_path(self):
        return str(self.result_path)
    
    @property
    def type(self):
        if self._media_type is None:
            self._media_type = LoaderMixin.get_media_type(self.path)
        return self._media_type
    
    def save_result(self, result: OutputMedia):
        if self.type == "image":
            result.save(self.result_path)
            
        elif self.type == "video":
            # Load existing cached frames if result already exists
            existing_frames = {}
            if self.result_path.exists() and self._metadata:
                result_video = decord.VideoReader(str(self.result_path))
                cached_frame_indices = sorted(self._metadata.get("cached_frames", []))
                if cached_frame_indices:
                    # Get all cached frames in a single batch
                    indices_to_read = list(range(min(len(cached_frame_indices), len(result_video))))
                    if indices_to_read:
                        frames_batch = result_video.get_batch(indices_to_read).asnumpy()
                        for idx, frame_idx in enumerate(cached_frame_indices[:len(indices_to_read)]):
                            existing_frames[frame_idx] = frames_batch[idx]
            
            # Merge existing cached frames with new frames
            frame_range = self._get_video_frame_range()
            merged_frames = []
            all_cached_frames = []
            
            for frame_idx in frame_range:
                if frame_idx in self.non_cached_frames:
                    # New frame from result
                    result_idx = self.non_cached_frames[frame_idx]
                    merged_frames.append(result[result_idx])
                    all_cached_frames.append(frame_idx)
                elif frame_idx in existing_frames:
                    # Previously cached frame
                    merged_frames.append(existing_frames[frame_idx])
                    all_cached_frames.append(frame_idx)
            
            # Save merged result
            if merged_frames:
                imageio.mimsave(self.result_path, merged_frames, fps=self._video_info["fps"])
                
                # Update metadata
                metadata = {
                    "cached_frames": sorted(all_cached_frames),
                    "frame_count": len(all_cached_frames),
                    "video_info": self._video_info
                }

                with open(self.metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
        return str(self.result_path)
    @property
    def image(self):
        return self._image
    
    @property
    def video(self):
        frame_range = self._get_video_frame_range()
        
        # Collect all non-cached frame indices
        non_cached_indices = [frame_idx for frame_idx in frame_range if frame_idx not in self.cached_frames]
        
        if not non_cached_indices:
            return []
        
        try:
            frames_batch = self._video.get_batch(non_cached_indices).asnumpy()
        except Exception as e:
            raise ValueError(f"Failed to read frames from video") from e
        
        # Map frame indices to output positions
        for idx, frame_idx in enumerate(non_cached_indices):
            self.non_cached_frames[frame_idx] = idx
        
        return list(frames_batch)