from sam2.build_sam import build_sam2_video_predictor
from sam2.sam2_video_predictor import SAM2VideoPredictor
from src.mixins import LoaderMixin
from PIL import Image
import torch
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Union, Iterator
from collections import OrderedDict
from loguru import logger
import traceback
# get the default device
from src.utils.defaults import get_torch_device, DEFAULT_PREPROCESSOR_SAVE_PATH

# create an enum for the model types 
from enum import Enum

class ModelType(Enum):
    SAM2_TINY = "sam2_tiny"
    SAM2_SMALL = "sam2_small"
    SAM2_BASE_PLUS = "sam2_base_plus"
    SAM2_LARGE = "sam2_large"
    

MODEL_WEIGHTS = {
    ModelType.SAM2_TINY: "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt",
    ModelType.SAM2_SMALL: "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt", 
    ModelType.SAM2_BASE_PLUS: "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt",
    ModelType.SAM2_LARGE: "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt",
}

MODEL_CONFIGS = {
    ModelType.SAM2_TINY: "configs/sam2.1/sam2.1_hiera_t",
    ModelType.SAM2_SMALL: "configs/sam2.1/sam2.1_hiera_s",
    ModelType.SAM2_BASE_PLUS: "configs/sam2.1/sam2.1_hiera_b+",
    ModelType.SAM2_LARGE: "configs/sam2.1/sam2.1_hiera_l",
}


def extract_video_frame(video_path: str, frame_number: int) -> np.ndarray:
    """
    Extract a specific frame from a video file using decord.
    
    Args:
        video_path: Path to the video file
        frame_number: Frame index to extract (0-based)
        
    Returns:
        numpy array of shape (H, W, 3) in RGB format
    """
    try:
        from decord import VideoReader, cpu
        
        # Use CPU context for frame extraction
        vr = VideoReader(video_path, ctx=cpu(0))
        
        # Validate frame number
        total_frames = len(vr)
        if frame_number < 0 or frame_number >= total_frames:
            raise ValueError(f"Frame number {frame_number} out of range [0, {total_frames-1}]")
        
        # Extract frame (decord returns RGB)
        frame = vr[frame_number].asnumpy()
        
        logger.info(f"Extracted frame {frame_number} from video: {video_path}, shape: {frame.shape}")
        return frame
        
    except ImportError:
        raise ImportError("decord is required for video frame extraction. Install with: pip install decord")
    except Exception as e:
        logger.error(f"Failed to extract frame from video: {str(e)}")
        raise


def mask_to_contours(
    mask: np.ndarray, 
    simplify_tolerance: float = 1.0,
    min_area: int = 100  # Filter out tiny contours
) -> List[List[float]]:
    """
    Convert a binary mask to contour polygon points (optimized for speed).
    
    Args:
        mask: Binary mask array of shape (H, W) with values 0 or 1
        simplify_tolerance: Epsilon value for contour simplification (Douglas-Peucker)
        min_area: Minimum contour area to keep (filters noise)
        
    Returns:
        List of contours, where each contour is a flat list [x1, y1, x2, y2, ...]
    """
    # Ensure mask is uint8 (optimize: avoid copy if possible)
    if mask.dtype != np.uint8:
        mask_uint8 = (mask * 255).astype(np.uint8)
    else:
        mask_uint8 = mask
    
    # Find contours with optimized method
    contours, _ = cv2.findContours(
        mask_uint8, 
        cv2.RETR_EXTERNAL,  # Only external contours
        cv2.CHAIN_APPROX_SIMPLE  # Compress segments
    )
    
    result_contours = []
    for contour in contours:
        # Filter small contours early (noise reduction)
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        
        # Simplify contour to reduce point count
        epsilon = simplify_tolerance
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Flatten to [x1, y1, x2, y2, ...] format
        # Optimize: use reshape(-1) instead of reshape(-1, 2).flatten()
        points = approx.reshape(-1).astype(np.float32).tolist()
        
        # Only include contours with at least 3 points (6 values)
        if len(points) >= 6:
            result_contours.append(points)
    
    logger.debug(f"Extracted {len(result_contours)} contours from mask (filtered {len(contours) - len(result_contours)} tiny contours)")
    return result_contours


class LazyFrameLoader:
    """
    Lazy frame loader that supports both images and videos.
    Loads frames on-demand to avoid memory issues with long videos.
    Caches VideoReader for the same video file.
    Applies SAM2 normalization (ImageNet mean/std).
    """
    
    # Class-level cache for VideoReader objects
    _video_reader_cache = {}
    _max_cached_readers = 2
    
    # SAM2 normalization constants (ImageNet)
    IMG_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)[:, None, None]
    IMG_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)[:, None, None]
    
    def __init__(self, input_path: str, frame_number: Optional[int] = None, image_size: int = 512):
        self.input_path = input_path
        self.frame_number = frame_number
        self.image_size = image_size
        
        path = Path(input_path)
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv', '.m4v'}
        self.is_video = path.suffix.lower() in video_extensions
        
        if self.is_video:
            # For videos, use cached decord VideoReader
            if input_path not in self._video_reader_cache:
                try:
                    from decord import VideoReader, cpu
                    
                    # Clean up old cached readers if we exceed max
                    if len(self._video_reader_cache) >= self._max_cached_readers:
                        oldest_key = next(iter(self._video_reader_cache))
                        del self._video_reader_cache[oldest_key]
                    
                    # Cache the VideoReader
                    vr = VideoReader(input_path, ctx=cpu(0))
                    self._video_reader_cache[input_path] = vr
                except ImportError:
                    raise ImportError("decord is required for video support. Install with: pip install decord")
            
            vr = self._video_reader_cache[input_path]
            
            # Get video dimensions from the first frame
            first_frame = vr[0].asnumpy()
            self.video_height, self.video_width = first_frame.shape[:2]
            
            # For now, we only support single frame
            self.num_frames = 1
            self.target_frame_idx = frame_number or 0
            
            # Validate frame number
            total_frames = len(vr)
            if self.target_frame_idx < 0:
                self.target_frame_idx = 0
            elif self.target_frame_idx >= total_frames:
                self.target_frame_idx = total_frames - 1
                
        else:
            # For images, treat as single frame
            self.num_frames = 1
            pil_image = Image.open(input_path).convert('RGB')
            self.video_width, self.video_height = pil_image.size
            self._cached_image = np.array(pil_image)
    
    def __len__(self):
        return self.num_frames
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Get a frame as a torch tensor, resized and normalized.
        
        Applies SAM2 normalization: (pixel/255.0 - mean) / std
        
        Returns:
            Tensor of shape (3, image_size, image_size), normalized
        """
        if idx >= self.num_frames:
            raise IndexError(f"Frame index {idx} out of range [0, {self.num_frames})")
        
        # Get the frame
        if self.is_video:
            vr = self._video_reader_cache[self.input_path]
            frame = vr[self.target_frame_idx].asnumpy()
        else:
            frame = self._cached_image
        
        # Convert to torch tensor (H, W, 3) -> (3, H, W)
        img_tensor = torch.from_numpy(frame).permute(2, 0, 1).float()
        
        # Resize to model's image_size
        img_resized = torch.nn.functional.interpolate(
            img_tensor.unsqueeze(0),
            size=(self.image_size, self.image_size),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)
        
        # Normalize: (pixel/255.0 - mean) / std (SAM2 ImageNet normalization)
        img_normalized = (img_resized / 255.0 - self.IMG_MEAN) / self.IMG_STD
        
        return img_normalized
    
    @classmethod
    def clear_cache(cls):
        """Clear all cached VideoReaders."""
        cls._video_reader_cache.clear()


class MultiFrameLoader:
    """
    Frame loader over a contiguous range of frames in a video.
    Provides normalized tensors compatible with SAM2 preprocessing.
    """
    _video_reader_cache = LazyFrameLoader._video_reader_cache  # share cache
    _max_cached_readers = LazyFrameLoader._max_cached_readers

    IMG_MEAN = LazyFrameLoader.IMG_MEAN
    IMG_STD = LazyFrameLoader.IMG_STD

    def __init__(
        self,
        input_path: str,
        frame_start: int,
        frame_end: int,
        image_size: int = 512,
        max_frames: Optional[int] = None,
    ):
        self.input_path = input_path
        self.image_size = image_size
        self.frame_start = frame_start
        self.frame_end = frame_end
        self.max_frames = max_frames

        if input_path not in self._video_reader_cache:
            try:
                from decord import VideoReader, cpu
                if len(self._video_reader_cache) >= self._max_cached_readers:
                    oldest_key = next(iter(self._video_reader_cache))
                    del self._video_reader_cache[oldest_key]
                vr = VideoReader(input_path, ctx=cpu(0))
                self._video_reader_cache[input_path] = vr
            except ImportError:
                raise ImportError("decord is required for video support. Install with: pip install decord")

        self.vr = self._video_reader_cache[input_path]

        total_frames = len(self.vr)
        anchor = max(0, min(self.frame_start, total_frames - 1))
        target = max(0, min(self.frame_end, total_frames - 1))

        low = min(anchor, target)
        high = max(anchor, target)

        if self.max_frames is not None and self.max_frames > 0:
            if target >= anchor:
                high = min(high, anchor + self.max_frames)
            else:
                low = max(low, anchor - self.max_frames)

        self.frame_indices = list(range(low, high + 1))
        self.anchor_idx = max(0, anchor - low)
        self.reverse = target < anchor
        self.anchor_frame = self.frame_indices[self.anchor_idx]
        self.target_frame = target
        self.frame_start = anchor
        self.frame_end = target

        # Resolve video dimensions from the first materialized frame
        first_frame = self.vr[self.anchor_frame].asnumpy()
        self.video_height, self.video_width = first_frame.shape[:2]
        
    def __len__(self) -> int:
        return len(self.frame_indices)

    def __getitem__(self, idx: int) -> torch.Tensor:
        if idx < 0 or idx >= len(self.frame_indices):
            raise IndexError(f"Frame index {idx} out of range [0, {len(self.frame_indices)})")
        frame_idx = self.frame_indices[idx]
        frame = self.vr[frame_idx].asnumpy()

        img_tensor = torch.from_numpy(frame).permute(2, 0, 1).float()
        img_resized = torch.nn.functional.interpolate(
            img_tensor.unsqueeze(0),
            size=(self.image_size, self.image_size),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)
        
        img_normalized = (img_resized / 255.0 - self.IMG_MEAN) / self.IMG_STD
        
        return img_normalized


class UnifiedSAM2VideoPredictor(SAM2VideoPredictor):
    """
    Custom SAM2VideoPredictor that supports both images and videos with lazy loading.
    Overrides init_state to use LazyFrameLoader instead of loading all frames.
    """
    
    @torch.inference_mode()
    def init_state(
        self,
        input_path: str,
        frame_number: Optional[int] = None,
        offload_video_to_cpu: bool = False,
        offload_state_to_cpu: bool = False,
    ):
        """
        Initialize an inference state with lazy frame loading.
        Supports both images and videos.
        
        Args:
            input_path: Path to image or video file
            frame_number: For videos, which frame to load (None for images)
            offload_video_to_cpu: Whether to offload video frames to CPU
            offload_state_to_cpu: Whether to offload inference state to CPU
        """
        compute_device = self.device
        
        # Create lazy frame loader
        images = LazyFrameLoader(input_path, frame_number, image_size=self.image_size)
        video_height = images.video_height
        video_width = images.video_width
        
        if self.device.type == "mps":
            offload_video_to_cpu = True
        
        # Initialize inference state (same structure as original)
        inference_state = {}
        inference_state["images"] = images
        inference_state["num_frames"] = len(images)
        inference_state["offload_video_to_cpu"] = offload_video_to_cpu
        inference_state["offload_state_to_cpu"] = offload_state_to_cpu
        inference_state["video_height"] = video_height
        inference_state["video_width"] = video_width
        inference_state["device"] = compute_device
        
        if offload_state_to_cpu:
            inference_state["storage_device"] = torch.device("cpu")
        else:
            inference_state["storage_device"] = compute_device
        
        # Initialize tracking structures
        inference_state["point_inputs_per_obj"] = {}
        inference_state["mask_inputs_per_obj"] = {}
        inference_state["cached_features"] = {}
        inference_state["constants"] = {}
        inference_state["obj_id_to_idx"] = OrderedDict()
        inference_state["obj_idx_to_id"] = OrderedDict()
        inference_state["obj_ids"] = []
        inference_state["output_dict_per_obj"] = {}
        inference_state["temp_output_dict_per_obj"] = {}
        inference_state["frames_tracked_per_obj"] = {}
        
        # Warm up the visual backbone and cache the image feature on frame 0
        self._get_image_feature(inference_state, frame_idx=0, batch_size=1)
        
        return inference_state


class UnifiedSAM2Predictor:
    """
    Unified predictor using custom UnifiedSAM2VideoPredictor for both images and videos.
    Images are treated as single-frame videos.
    Uses lazy frame loading to handle long videos efficiently without temp directories.
    
    Performance optimizations:
    - Half precision (FP16/BF16) inference for 2x speedup
    - Torch compile for optimized execution
    - TF32 tensor cores on Ampere+ GPUs
    - Aggressive feature caching
    """
    
    def __init__(
        self, 
        model_path: str, 
        config_name: str, 
        model_type: ModelType = ModelType.SAM2_BASE_PLUS,
        dtype_str: str = "float32",  # Pass as string to avoid torch pickling issues
        use_compile: bool = False,
        use_tf32: bool = True,
    ):
        # Create logger inside actor to avoid pickling issues
        from loguru import logger as actor_logger
        self.logger = actor_logger
        
        self.model_type = model_type
        self.model_path = model_path
        self.config_name = config_name
        self.use_compile = use_compile
        self.use_tf32 = use_tf32
        
        # Get device inside actor
        self.device = get_torch_device()

        # Convert dtype string to torch dtype
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        self.dtype = dtype_map.get(dtype_str, torch.float32)
        
        # Single video predictor for everything
        self._predictor = None
        
        # Cache inference states by input hash
        # Format: {(input_path, frame_number): inference_state}
        self._inference_states = {}
        self._max_cached_states = 5  # Increased from 3 for better caching
        
        self.logger.info(f"UnifiedSAM2Predictor initialized with config: {config_name}, "
                   f"device: {self.device}, dtype: {self.dtype}, compile: {use_compile}")
    
    def get_predictor(self) -> UnifiedSAM2VideoPredictor:
        """Get or create the custom video predictor with optimizations."""
        if self._predictor is None:
            # Enable TF32 for Ampere+ GPUs (must be done before model load)
            if self.use_tf32:
                try:
                    import torch.backends.cuda
                    import torch.backends.cudnn
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.allow_tf32 = True
                    self.logger.info("Enabled TF32 tensor cores for faster inference")
                except:
                    pass  # Not on CUDA or TF32 not available
            
            self.logger.info(f"Loading unified SAM2 video predictor with config: {self.config_name}, checkpoint: {self.model_path}")
            
            # Build the predictor
            predictor = build_sam2_video_predictor(
                self.config_name, 
                self.model_path, 
                device=self.device
            )
            
            # Apply half precision if enabled
            if self.dtype != torch.float32:
                predictor = predictor.to(dtype=self.dtype)
                self.logger.info(f"Converted model to {self.dtype}")
            
            # Apply torch.compile if enabled (PyTorch 2.0+)
            if self.use_compile:
                try:
                    self.logger.info("Compiling model with torch.compile (first run will be slow)...")
                    # Compile the model forward pass
                    predictor.forward_image = torch.compile(
                        predictor.forward_image, 
                        mode="reduce-overhead"  # Best for repeated calls
                    )
                    self.logger.info("Model compiled successfully")
                except Exception as e:
                    self.logger.warning(f"Failed to compile model: {e}. Continuing without compilation.")
            
            # Convert to our custom subclass by copying attributes
            custom_predictor = UnifiedSAM2VideoPredictor.__new__(UnifiedSAM2VideoPredictor)
            custom_predictor.__dict__.update(predictor.__dict__)
            self._predictor = custom_predictor
            
            # Pre-warm the model with a dummy forward pass
            self._warmup_model()
            
            self.logger.info("Unified SAM2 video predictor loaded and optimized")
        return self._predictor
    
    def _warmup_model(self):
        """Pre-warm the model to compile CUDA kernels."""
        try:
            self.logger.info("Warming up model...")
            dummy_img = torch.rand(1, 3, 1024, 1024, device=self.device, dtype=self.dtype)
            with torch.inference_mode():
                _ = self._predictor.forward_image(dummy_img)
            self.logger.info("Model warmup complete")
        except Exception as e:
            self.logger.warning(f"Warmup failed: {e}. Model will warm up on first real inference.")
    
    def _get_cache_key(self, input_path: str, frame_number: Optional[int], id: Optional[str] = None) -> Tuple[str, Optional[int], Optional[str]]:
        """Generate cache key for inference state."""
        return (input_path, frame_number, id)
    
    def _cleanup_oldest_state(self):
        """Remove the oldest cached inference state to manage memory."""
        if len(self._inference_states) >= self._max_cached_states:
            # Remove oldest entry (FIFO)
            oldest_key = next(iter(self._inference_states))
            del self._inference_states[oldest_key]
            self.logger.debug(f"Evicted oldest inference state: {oldest_key}")
    
    def _get_or_create_inference_state(self, input_path: str, frame_number: Optional[int] = None, id: Optional[str] = None):
        """Get cached inference state or create new one."""
        cache_key = self._get_cache_key(input_path, frame_number, id)
        
        if cache_key in self._inference_states:
            self.logger.debug(f"Using cached inference state for {cache_key}")
            return self._inference_states[cache_key]
        
        # Cleanup old states if needed
        self._cleanup_oldest_state()
        
        # Initialize inference state using our custom init_state (no temp dir needed!)
        predictor = self.get_predictor()
        
        inference_state = predictor.init_state(
            input_path=input_path,
            frame_number=frame_number,
            offload_video_to_cpu=False,  # Keep in GPU for speed
            offload_state_to_cpu=False    # Keep in GPU for speed
        )
        
        # Cache it
        self._inference_states[cache_key] = inference_state
        
        return inference_state
    
    def _cache_inference_state(
        self,
        inference_state: dict,
        input_path: str,
        frame_number: Optional[int] = None,
        id: Optional[str] = None,
        prompts: Optional[dict] = None,
    ):
        """Cache the inference state along with prompt metadata for reuse."""
        cache_key = self._get_cache_key(input_path, frame_number, id)

        if prompts:
            state_prompts = inference_state.setdefault("cached_prompts", {})
            state_prompts[id or "default"] = prompts
            

        self._inference_states[cache_key] = inference_state

    def _maybe_restore_state(
        self,
        input_path: str,
        frame_start: int,
        id: Optional[str],
    ):
        cache_key = self._get_cache_key(input_path, frame_start, id)
        if cache_key not in self._inference_states:
            return self._get_or_create_inference_state(input_path, frame_start, id)

        inference_state = self._inference_states[cache_key]
        # Prompts are already embedded in cached inference state; nothing additional needed.

        return inference_state

    def _remap_cached_frame_index(self, inference_state: dict, old_idx: int, new_idx: int) -> None:
        """Remap all per-frame caches from old_idx to new_idx after replacing images loader.

        This aligns the conditioning frame (created at index 0 in single-frame mode)
        with the anchor index in the multi-frame loader so tracking behaves correctly
        in both forward and reverse directions.
        """
        if old_idx == new_idx:
            return

        try:
            obj_indices = list(inference_state["obj_idx_to_id"].keys())
        except Exception:
            obj_indices = []

        # Remap per-object inputs and outputs
        for obj_idx in obj_indices:
            # point inputs
            point_map = inference_state["point_inputs_per_obj"].get(obj_idx, {})
            if old_idx in point_map and new_idx not in point_map:
                point_map[new_idx] = point_map.pop(old_idx)

            # mask inputs
            mask_map = inference_state["mask_inputs_per_obj"].get(obj_idx, {})
            if old_idx in mask_map and new_idx not in mask_map:
                mask_map[new_idx] = mask_map.pop(old_idx)

            # output dicts (consolidated)
            out_dict = inference_state["output_dict_per_obj"].get(obj_idx, {})
            for storage_key in ("cond_frame_outputs", "non_cond_frame_outputs"):
                frames_map = out_dict.get(storage_key, {})
                if old_idx in frames_map and new_idx not in frames_map:
                    frames_map[new_idx] = frames_map.pop(old_idx)

            # temp outputs (most recent)
            temp_out_dict = inference_state["temp_output_dict_per_obj"].get(obj_idx, {})
            for storage_key in ("cond_frame_outputs", "non_cond_frame_outputs"):
                frames_map = temp_out_dict.get(storage_key, {})
                if old_idx in frames_map and new_idx not in frames_map:
                    frames_map[new_idx] = frames_map.pop(old_idx)

            # frames tracked metadata
            tracked_map = inference_state["frames_tracked_per_obj"].get(obj_idx, {})
            if old_idx in tracked_map and new_idx not in tracked_map:
                tracked_map[new_idx] = tracked_map.pop(old_idx)

    @torch.inference_mode()
    def predict_mask(
        self,
        input_path: str,
        frame_number: Optional[int] = None,
        point_coords: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
        box: Optional[np.ndarray] = None,
        obj_id: int = 1,
        simplify_tolerance: float = 1.0,
        id: Optional[str] = None
    ) -> Tuple[List[List[float]], np.ndarray]:
        """
        Generate mask using video predictor (works for both images and videos).
        
        Args:
            input_path: Path to image or video file
            frame_number: Frame index for videos (None for images)
            point_coords: Array of shape (N, 2) with point coordinates
            point_labels: Array of shape (N,) with point labels (1=positive, 0=negative)
            box: Array of shape (4,) with box coordinates [x1, y1, x2, y2]
            obj_id: Object ID for tracking (default 1)
            simplify_tolerance: Contour simplification tolerance
            
        Returns:
            Tuple of (contours, mask) where contours is a list of polygon point lists
        """
        predictor = self.get_predictor()
        
        self.logger.info(f"Predicting mask for {input_path}, frame_number: {frame_number}, id: {id}")
        
        # Get or create inference state
        inference_state = self._maybe_restore_state(
            input_path=input_path,
            frame_start=frame_number or 0,
            id=id,
        )
        
        self.logger.info(f"Inference states: {len(inference_state['obj_idx_to_id'])}")
        
        # We always use frame 0 since we load only the specific frame
        target_frame_idx = 0
        
        # Convert numpy arrays to lists for SAM2's API
        points = point_coords.tolist() if point_coords is not None else None
        labels = point_labels.tolist() if point_labels is not None else None
        box_list = box.tolist() if box is not None else None
        
        self.logger.info(f"points: {points}, labels: {labels}, box: {box_list}")
        
        # Add prompts and get mask
        self.logger.debug(f"Adding prompts: points={len(points) if points else 0}, "
                    f"box={'yes' if box_list else 'no'}")
        
        frame_idx, obj_ids, video_res_masks = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=target_frame_idx,
            obj_id=obj_id,
            points=points,
            labels=labels,
            box=box_list,
            normalize_coords=True
        )
        
        # Extract mask for our object (first object, first channel)
        mask = video_res_masks[0, 0].cpu().numpy()  # (H, W)
        
        # Convert mask scores to binary
        mask_binary = (mask > 0.0).astype(np.uint8)
        
        # Convert to contours
        contours = mask_to_contours(mask_binary, simplify_tolerance=simplify_tolerance)
        
        self.logger.info(f"Generated mask with {len(contours)} contours for object {obj_id}")

        prompt_metadata = {
            "points": points,
            "labels": labels,
            "box": box_list,
            "obj_id": obj_id,
        }

        self._cache_inference_state(
            inference_state,
            input_path,
            frame_number,
            id,
            prompts=prompt_metadata,
        )
        return contours, mask_binary
    
    @torch.inference_mode()
    def track_masks(
        self,
        input_path: str,
        frame_start: int,
        frame_end: int,
        anchor_frame: Optional[int] = None,
        max_frames: Optional[int] = None,
        id: Optional[str] = None,
        simplify_tolerance: float = 1.0,
    ) -> List[dict]:
        """
        Propagate the existing object mask from the cached inference state across a frame range.

        Returns a list of per-frame dicts: {"frame_number": int, "contours": List[List[float]]}
        """
        predictor = self.get_predictor()

        inference_state = self._maybe_restore_state(
            input_path=input_path,
            frame_start=frame_start,
            id=id,
        )
        
        # Swap in a multi-frame loader spanning the requested range
        loader = MultiFrameLoader(
            input_path=input_path,
            frame_start=anchor_frame if anchor_frame is not None else frame_start,
            frame_end=frame_end,
            image_size=predictor.image_size,
            max_frames=max_frames,
        )
        inference_state["images"] = loader
        inference_state["num_frames"] = len(loader)
        inference_state["video_height"] = loader.video_height
        inference_state["video_width"] = loader.video_width
        # Remap any cached frame index from previous single-frame session (0) to anchor idx
        self._remap_cached_frame_index(inference_state, old_idx=0, new_idx=loader.anchor_idx)
        # Clear cached features as images changed, warm up on anchor frame
        inference_state["cached_features"] = {}
        predictor._get_image_feature(inference_state, frame_idx=loader.anchor_idx, batch_size=1)

        max_frames = len(loader)
        reverse = loader.reverse
        results: List[dict] = []
        
        
        
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
            inference_state,
            start_frame_idx=loader.anchor_idx,
            max_frame_num_to_track=max_frames,
            reverse=reverse,
        ):
            abs_frame = loader.frame_indices[out_frame_idx]

            frame_contours: List[List[float]] = []
            for i, _obj_id in enumerate(out_obj_ids):
                mask = (out_mask_logits[i] > 0.0).detach().cpu().numpy().squeeze(0)
                contours = mask_to_contours(mask.astype(np.uint8), simplify_tolerance=simplify_tolerance)
                if contours:
                    frame_contours.extend(contours)

            results.append({
                "frame_number": int(abs_frame),
                "contours": frame_contours,
            })

        return results

    @torch.inference_mode()
    def iter_track_masks(
        self,
        input_path: str,
        frame_start: int,
        frame_end: int,
        anchor_frame: Optional[int] = None,
        max_frames: Optional[int] = None,
        id: Optional[str] = None,
        simplify_tolerance: float = 1.0,
    ) -> Iterator[dict]:
        """
        Yield propagated mask contours per frame across a frame range.
        """
        predictor = self.get_predictor()

        inference_state = self._maybe_restore_state(
            input_path=input_path,
            frame_start=frame_start,
            id=id,
        )

        loader = MultiFrameLoader(
            input_path=input_path,
            frame_start=anchor_frame if anchor_frame is not None else frame_start,
            frame_end=frame_end,
            image_size=predictor.image_size,
            max_frames=max_frames,
        )
        inference_state["images"] = loader
        inference_state["num_frames"] = len(loader)
        inference_state["video_height"] = loader.video_height
        inference_state["video_width"] = loader.video_width
        # Remap any cached frame index from previous single-frame session (0) to anchor idx
        self._remap_cached_frame_index(inference_state, old_idx=0, new_idx=loader.anchor_idx)
        # Clear cached features as images changed, warm up on anchor frame
        inference_state["cached_features"] = {}
        predictor._get_image_feature(inference_state, frame_idx=loader.anchor_idx, batch_size=1)

        total = max(1, len(loader))
        reverse = loader.reverse
        
        logger.info(f"{loader.anchor_idx} {loader.reverse} {max_frames}")


        try:
            for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
                inference_state,
                start_frame_idx=loader.anchor_idx,
                max_frame_num_to_track=total,
                reverse=reverse,
            ):
                abs_frame = loader.frame_indices[out_frame_idx]
                frame_contours: List[List[float]] = []
                for i, _obj_id in enumerate(out_obj_ids):
                    mask = (out_mask_logits[i] > 0.0).detach().cpu().numpy().squeeze(0)
                    contours = mask_to_contours(mask.astype(np.uint8), simplify_tolerance=simplify_tolerance)
                    if contours:
                        frame_contours.extend(contours)

                yield {
                    "frame_number": int(abs_frame),
                    "contours": frame_contours,
                }
        except Exception as e:
            logger.error(traceback.format_exc())
            raise e

    def cleanup(self):
        """Clean up all cached inference states and VideoReaders."""
        self._inference_states.clear()
        LazyFrameLoader.clear_cache()
        self.logger.info("Cleared all cached inference states and VideoReaders")


_PREDICTOR_SINGLETONS: dict = {}


def get_sam2_predictor(
    model_type: ModelType = ModelType.SAM2_SMALL,
    use_half_precision: bool = True,
    use_compile: bool = False,
    use_tf32: bool = True,
) -> UnifiedSAM2Predictor:
    """
    Get or create a singleton UnifiedSAM2Predictor in-process.
    """
    key = (model_type.value, use_half_precision, use_compile, use_tf32)
    if key in _PREDICTOR_SINGLETONS:
        return _PREDICTOR_SINGLETONS[key]

    logger.info("Creating new unified SAM2 predictor (in-process)")
    loader = LoaderMixin()
    model_path = loader._download(MODEL_WEIGHTS[model_type], save_path=DEFAULT_PREPROCESSOR_SAVE_PATH)
    config_name = MODEL_CONFIGS[model_type]

    dtype_str = "float32"
    if use_half_precision:
        device = get_torch_device()
        if device.type == "cuda":
            try:
                if torch.cuda.is_bf16_supported():
                    dtype_str = "bfloat16"
                    logger.info("Will use BFloat16 precision")
                else:
                    dtype_str = "float16"
                    logger.info("Will use Float16 precision")
            except Exception:
                dtype_str = "float16"
                logger.info("Will use Float16 precision")
        else:
            logger.info("Half precision not supported on non-CUDA device, using Float32")

    predictor = UnifiedSAM2Predictor(
        model_path=model_path,
        config_name=config_name,
        model_type=model_type,
        dtype_str=dtype_str,
        use_compile=use_compile,
        use_tf32=use_tf32,
    )

    _PREDICTOR_SINGLETONS[key] = predictor
    return predictor


        
