"""
Modern DepthAnything implementation using HuggingFace transformers.
Replaces legacy torch.hub.load DINOv2 backbone with transformers pipeline.
"""

import numpy as np
import torch
from PIL import Image
from transformers import pipeline
from pathlib import Path

from src.auxillary.util import HWC3, resize_image_with_pad
from src.utils.defaults import get_torch_device
from src.mixins import ToMixin
from src.types import InputImage, OutputImage
from src.auxillary.base_preprocessor import BasePreprocessor
from src.utils.defaults import DEFAULT_PREPROCESSOR_SAVE_PATH

class DepthAnythingDetector(ToMixin, BasePreprocessor):
    """DepthAnything depth estimation using HuggingFace transformers."""
    
    def __init__(self, model_name="LiheYoung/depth-anything-large-hf", cache_dir=None):
        """Initialize DepthAnything with specified model."""
        super().__init__()
        if cache_dir is None:
            cache_dir = Path(DEFAULT_PREPROCESSOR_SAVE_PATH) / "depth_anything"
        
        self.pipe = pipeline(
            task="depth-estimation", 
            model=model_name,
            model_kwargs={"cache_dir": str(cache_dir)},
            device=get_torch_device()
        )
        self.device = get_torch_device()

    @classmethod  
    def from_pretrained(cls, pretrained_model_or_path=None, filename="depth_anything_vitl14.pth"):
        """Create DepthAnything from pretrained model, mapping legacy names to HuggingFace models."""
        
        # Map legacy checkpoint names to modern HuggingFace models
        model_mapping = {
            "depth_anything_vitl14.pth": "LiheYoung/depth-anything-large-hf",
            "depth_anything_vitb14.pth": "LiheYoung/depth-anything-base-hf", 
            "depth_anything_vits14.pth": "LiheYoung/depth-anything-small-hf"
        }
        
        model_name = model_mapping.get(filename, "LiheYoung/depth-anything-large-hf")
        cache_dir = Path(DEFAULT_PREPROCESSOR_SAVE_PATH) / "depth_anything"
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cls(model_name=model_name, cache_dir=cache_dir)
    
    def process(self, input_image: InputImage, detect_resolution=512, upscale_method="INTER_CUBIC", **kwargs) -> OutputImage:
        """Perform depth estimation on input image."""
        input_image = self._load_image(input_image)
        
        if not isinstance(input_image, np.ndarray):
            input_image = np.asarray(input_image, dtype=np.uint8)
        
        input_image, remove_pad = resize_image_with_pad(input_image, detect_resolution, upscale_method)
        
        if isinstance(input_image, np.ndarray):
            pil_image = Image.fromarray(input_image)
        else:
            pil_image = input_image
        
        with torch.no_grad():
            result = self.pipe(pil_image)
            depth = result["depth"]
            
            if isinstance(depth, Image.Image):
                depth_array = np.array(depth, dtype=np.float32)
            else:
                depth_array = np.array(depth)
                
            # Normalize depth values to 0-255 range
            depth_min = depth_array.min()
            depth_max = depth_array.max()
            if depth_max > depth_min:
                depth_array = (depth_array - depth_min) / (depth_max - depth_min) * 255.0
            else:
                depth_array = np.zeros_like(depth_array)
                
            depth_image = depth_array.astype(np.uint8)

        detected_map = remove_pad(HWC3(depth_image))
        detected_map = Image.fromarray(detected_map)
        
        return detected_map