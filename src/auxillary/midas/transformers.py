"""
MiDaS implementation using HuggingFace transformers for PyTorch 2.7 compatibility.
"""
import numpy as np
import torch
import cv2
from PIL import Image
from typing import Union
from pathlib import Path

# Import utilities
from src.auxillary.util import HWC3, resize_image_with_pad
from src.mixins import ToMixin
from src.utils.defaults import get_torch_device, DEFAULT_PREPROCESSOR_SAVE_PATH
from src.types import InputImage, OutputImage
from src.auxillary.base_preprocessor import BasePreprocessor


class MidasDetector(ToMixin, BasePreprocessor):
    
    def __init__(self, model_name="Intel/dpt-large", cache_dir=None):
        from transformers import DPTForDepthEstimation, DPTImageProcessor
        
        super().__init__()
        if cache_dir is None:
            cache_dir = Path(DEFAULT_PREPROCESSOR_SAVE_PATH) / "midas"
            cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.model_name = model_name
        self.processor = DPTImageProcessor.from_pretrained(model_name, cache_dir=str(cache_dir))
        self.model = DPTForDepthEstimation.from_pretrained(model_name, cache_dir=str(cache_dir))
        self.device = get_torch_device()
        self.to_device(self.model, device=self.device)

    @classmethod  
    def from_pretrained(cls, pretrained_model_or_path=None, model_type="dpt_hybrid", filename="dpt_hybrid-midas-501f0c75.pt"):
        # Map legacy model types to HuggingFace models
        model_mapping = {
            "dpt_large": "Intel/dpt-large",
            "dpt_hybrid": "Intel/dpt-hybrid-midas", 
            "midas_v21": "Intel/dpt-large",
            "midas_v21_small": "Intel/dpt-large"
        }
        
        # Use filename for model selection if provided
        if filename and isinstance(filename, str):
            if "dpt_large" in filename.lower():
                model_name = "Intel/dpt-large"
            elif "dpt_hybrid" in filename.lower():
                model_name = "Intel/dpt-hybrid-midas"
            else:
                model_name = model_mapping.get(model_type, "Intel/dpt-large")
        else:
            model_name = model_mapping.get(model_type, "Intel/dpt-large")
        
        cache_dir = Path(DEFAULT_PREPROCESSOR_SAVE_PATH) / "midas"
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cls(model_name, cache_dir=cache_dir)

    def process(self, input_image: InputImage, a=np.pi * 2.0, bg_th=0.1, depth_and_normal=False, detect_resolution=512, upscale_method="INTER_CUBIC", **kwargs) -> OutputImage:
        input_image = self._load_image(input_image)
        
        if not isinstance(input_image, np.ndarray):
            input_image = np.array(input_image, dtype=np.uint8)
        
        detected_map, remove_pad = resize_image_with_pad(input_image, detect_resolution, upscale_method)
        
        # Convert to PIL for processor
        pil_image = Image.fromarray(detected_map.astype(np.uint8))
        
        # Process with HuggingFace pipeline
        with torch.no_grad():
            inputs = self.processor(images=pil_image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            outputs = self.model(**inputs)
            depth = outputs.predicted_depth
            
            # Normalize depth
            depth = torch.nn.functional.interpolate(
                depth.unsqueeze(1),
                size=(detected_map.shape[0], detected_map.shape[1]),
                mode="bicubic",
                align_corners=False,
            ).squeeze()
            
            depth_pt = depth.clone()
            depth_pt -= torch.min(depth_pt)
            depth_pt /= torch.max(depth_pt)
            depth_pt = depth_pt.cpu().numpy()
            depth_image = (depth_pt * 255.0).clip(0, 255).astype(np.uint8)

            if depth_and_normal:
                depth_np = depth.cpu().numpy()
                x = cv2.Sobel(depth_np, cv2.CV_32F, 1, 0, ksize=3)
                y = cv2.Sobel(depth_np, cv2.CV_32F, 0, 1, ksize=3)
                z = np.ones_like(x) * a
                x[depth_pt < bg_th] = 0
                y[depth_pt < bg_th] = 0
                normal = np.stack([x, y, z], axis=2)
                normal /= np.sum(normal ** 2.0, axis=2, keepdims=True) ** 0.5
                normal_image = (normal * 127.5 + 127.5).clip(0, 255).astype(np.uint8)[:, :, ::-1]
        
        depth_image = HWC3(depth_image)
        if depth_and_normal:
            normal_image = HWC3(normal_image)

        depth_image = remove_pad(depth_image)
        if depth_and_normal:
            normal_image = remove_pad(normal_image)
        
        depth_image = depth_image.astype(np.uint8)
        depth_image = Image.fromarray(depth_image)
        if depth_and_normal:
            normal_image = normal_image.astype(np.uint8)
            normal_image = Image.fromarray(normal_image)
        
        if depth_and_normal:
            return depth_image, normal_image
        else:
            return depth_image