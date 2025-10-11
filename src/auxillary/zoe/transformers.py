"""
ZoeDepth implementation using HuggingFace transformers.
Uses official Intel models for depth estimation.
"""

import numpy as np
import torch
from PIL import Image
from transformers import pipeline, AutoImageProcessor, ZoeDepthForDepthEstimation
from src.auxillary.util import resize_image_with_pad, HWC3
from src.types import InputImage, OutputImage
from src.auxillary.base_preprocessor import BasePreprocessor
from src.mixins import ToMixin
from src.utils.defaults import get_torch_device

class ZoeDetector(ToMixin, BasePreprocessor):
    """ZoeDepth depth estimation using HuggingFace transformers."""
    
    def __init__(self, model_name="Intel/zoedepth-nyu-kitti"):
        """Initialize ZoeDepth with specified model."""
        super().__init__()
        self.device = get_torch_device()
        self.pipe = pipeline(task="depth-estimation", model=model_name, device=self.device)

    @classmethod  
    def from_pretrained(cls, pretrained_model_or_path="Intel/zoedepth-nyu-kitti", filename=None, **kwargs):
        """Create ZoeDetector from pretrained model."""
        return cls(model_name=pretrained_model_or_path)
        
    def process(self, input_image: InputImage, detect_resolution=512, upscale_method="INTER_CUBIC", **kwargs) -> OutputImage:
        """Perform depth estimation on input image."""
        input_image = self._load_image(input_image)
        
        if not isinstance(input_image, np.ndarray):
            input_image = np.array(input_image, dtype=np.uint8)
        
        input_image, remove_pad = resize_image_with_pad(input_image, detect_resolution, upscale_method)
        
        pil_image = Image.fromarray(input_image)
        
        with torch.no_grad():
            result = self.pipe(pil_image)
            depth = result["depth"]
            
            if isinstance(depth, Image.Image):
                depth_array = np.array(depth, dtype=np.float32)
            else:
                depth_array = np.array(depth)
                
            vmin = np.percentile(depth_array, 2)
            vmax = np.percentile(depth_array, 85)
            
            depth_array = depth_array - vmin
            depth_array = depth_array / (vmax - vmin)
            depth_array = 1.0 - depth_array
            depth_image = (depth_array * 255.0).clip(0, 255).astype(np.uint8)

        detected_map = remove_pad(HWC3(depth_image))
        
        detected_map = Image.fromarray(detected_map)
            
        return detected_map


class ZoeDepthAnythingDetector(ToMixin, BasePreprocessor):
    """ZoeDepthAnything implementation using HuggingFace transformers."""
    
    def __init__(self, model_name="Intel/zoedepth-nyu-kitti"):
        """Initialize ZoeDepthAnything detector."""
        super().__init__()
        self.pipe = pipeline(task="depth-estimation", model=model_name)
        self.device = "cpu"

    @classmethod  
    def from_pretrained(cls, pretrained_model_or_path="Intel/zoedepth-nyu-kitti", filename=None, **kwargs):
        """Create from pretrained model."""
        return cls(model_name=pretrained_model_or_path)
        
    def process(self, input_image: InputImage, detect_resolution=512, upscale_method="INTER_CUBIC", **kwargs) -> OutputImage:
        """Perform depth estimation."""
        input_image = self._load_image(input_image)
        
        if not isinstance(input_image, np.ndarray):
            input_image = np.array(input_image, dtype=np.uint8)
        
        input_image, remove_pad = resize_image_with_pad(input_image, detect_resolution, upscale_method)
        
        pil_image = Image.fromarray(input_image)
        
        with torch.no_grad():
            result = self.pipe(pil_image)
            depth = result["depth"]
            
            if isinstance(depth, Image.Image):
                depth_array = np.array(depth, dtype=np.float32)
            else:
                depth_array = np.array(depth)
                
            vmin = np.percentile(depth_array, 2)
            vmax = np.percentile(depth_array, 85)
            
            depth_array = depth_array - vmin
            depth_array = depth_array / (vmax - vmin)
            depth_array = 1.0 - depth_array
            depth_image = (depth_array * 255.0).clip(0, 255).astype(np.uint8)

        detected_map = remove_pad(HWC3(depth_image))
        
        detected_map = Image.fromarray(detected_map)
            
        return detected_map