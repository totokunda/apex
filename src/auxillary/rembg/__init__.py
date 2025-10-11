"""
Rembg background removal preprocessor for image and video processing
Uses the rembg library for high-quality background removal
"""
import os
import warnings
from typing import Optional

import cv2
import numpy as np
from PIL import Image

from src.auxillary.util import HWC3, resize_image_with_pad, custom_hf_download
from src.types import InputImage, InputVideo, OutputImage, OutputVideo
from src.auxillary.base_preprocessor import BasePreprocessor
from src.utils.defaults import DEFAULT_PREPROCESSOR_SAVE_PATH


class RembgDetector(BasePreprocessor):
    """
    Rembg background removal detector for image and video processing.
    Uses the rembg library with various model options.
    """
    
    def __init__(self, model_name: str = "u2net"):
        super().__init__()
        self.model_name = model_name
        self.session = None
    
    @classmethod
    def from_pretrained(cls, pretrained_model_or_path: str = "u2net", **kwargs):
        """
        Load Rembg model from pretrained weights.
        
        Args:
            pretrained_model_or_path: Model name from rembg 
                                     ("u2net", "u2netp", "u2net_human_seg", "silueta", "isnet-general-use")
            **kwargs: Additional arguments for model initialization
            
        Returns:
            RembgDetector instance
        """
        try:
            # Import rembg
            from rembg import new_session
        except ImportError:
            raise ImportError(
                "rembg is not installed. Please install it with: pip install rembg"
            )
        
        model_name = pretrained_model_or_path
        
        try:
            # Set rembg cache directory to our preprocessor save path
            rembg_cache = os.path.join(DEFAULT_PREPROCESSOR_SAVE_PATH, "rembg")
            os.makedirs(rembg_cache, exist_ok=True)
            
            # Set environment variable for rembg to use our cache directory
            original_u2net_home = os.environ.get('U2NET_HOME')
            os.environ['U2NET_HOME'] = rembg_cache
            
            try:
                # Create a new rembg session with the specified model
                # This will download the model if not already cached
                instance = cls(model_name)
                instance.session = new_session(model_name)
                
            finally:
                # Restore original U2NET_HOME if it was set
                if original_u2net_home is not None:
                    os.environ['U2NET_HOME'] = original_u2net_home
                elif 'U2NET_HOME' in os.environ:
                    del os.environ['U2NET_HOME']
            
        except Exception as e:
            raise ValueError(
                f"Failed to load rembg model '{model_name}'. "
                f"Error: {str(e)}. "
                f"Available models: u2net, u2netp, u2net_human_seg, silueta, isnet-general-use"
            )
        
        return instance
    
    def process(self, input_image: InputImage, alpha_matting=False, alpha_matting_foreground_threshold=240,
                alpha_matting_background_threshold=10, alpha_matting_erode_size=10,
                post_process_mask=False, bgcolor=None, detect_resolution=0, 
                upscale_method="INTER_CUBIC", **kwargs) -> OutputImage:
        """
        Process a single image to remove background.
        
        Args:
            input_image: Input image
            alpha_matting: Enable alpha matting for better edges
            alpha_matting_foreground_threshold: Foreground threshold for alpha matting
            alpha_matting_background_threshold: Background threshold for alpha matting
            alpha_matting_erode_size: Erosion size for alpha matting
            post_process_mask: Apply post-processing to the mask
            bgcolor: Background color tuple (R, G, B, A) or None for transparent
            detect_resolution: Resolution for processing (0 = original size)
            upscale_method: Method for resizing
            **kwargs: Additional processing parameters
            
        Returns:
            Image with background removed (RGBA)
        """
        from rembg import remove
        
        input_image = self._load_image(input_image)
        
        if not isinstance(input_image, np.ndarray):
            input_image = np.array(input_image, dtype=np.uint8)
        
        # Convert to PIL Image for rembg
        if input_image.shape[2] == 4:
            pil_image = Image.fromarray(input_image, mode='RGBA')
        else:
            pil_image = Image.fromarray(input_image, mode='RGB')
        
        # Resize if needed
        if detect_resolution > 0:
            input_resized, remove_pad = resize_image_with_pad(input_image, detect_resolution, upscale_method)
            pil_image = Image.fromarray(input_resized)
        else:
            remove_pad = lambda x: x
        
        # Remove background using rembg
        output = remove(
            pil_image,
            session=self.session,
            alpha_matting=alpha_matting,
            alpha_matting_foreground_threshold=alpha_matting_foreground_threshold,
            alpha_matting_background_threshold=alpha_matting_background_threshold,
            alpha_matting_erode_size=alpha_matting_erode_size,
            post_process_mask=post_process_mask,
            bgcolor=bgcolor
        )
        
        # Convert back to numpy for padding removal if needed
        if detect_resolution > 0:
            output_np = np.array(output)
            output_np = remove_pad(output_np)
            output = Image.fromarray(output_np)
        
        return output
    
    def process_video(self, input_video: InputVideo, alpha_matting=False, 
                     alpha_matting_foreground_threshold=240,
                     alpha_matting_background_threshold=10, 
                     alpha_matting_erode_size=10,
                     post_process_mask=False, bgcolor=None,
                     detect_resolution=0, upscale_method="INTER_CUBIC", **kwargs) -> OutputVideo:
        """
        Process video to remove background from all frames.
        
        Args:
            input_video: Input video as list of frames or video path
            alpha_matting: Enable alpha matting for better edges
            alpha_matting_foreground_threshold: Foreground threshold for alpha matting
            alpha_matting_background_threshold: Background threshold for alpha matting
            alpha_matting_erode_size: Erosion size for alpha matting
            post_process_mask: Apply post-processing to the mask
            bgcolor: Background color tuple (R, G, B, A) or None for transparent
            detect_resolution: Resolution for processing (0 = original size)
            upscale_method: Method for resizing
            **kwargs: Additional processing parameters (including progress_callback)
            
        Returns:
            List of images with background removed
        """
        from tqdm import tqdm
        
        frames = self._load_video(input_video)
        
        ret_frames = []
        
        # Get progress callback if provided
        progress_callback = kwargs.get("progress_callback", None)
        total_frames = len(frames)
        
        # Process each frame
        for frame_idx in tqdm(range(total_frames), desc="Removing background"):
            # Update progress
            if progress_callback is not None:
                progress_callback(frame_idx + 1, total_frames)
            
            # Get frame
            frame = frames[frame_idx]
            
            # Process the frame
            result = self.process(
                frame, 
                alpha_matting=alpha_matting,
                alpha_matting_foreground_threshold=alpha_matting_foreground_threshold,
                alpha_matting_background_threshold=alpha_matting_background_threshold,
                alpha_matting_erode_size=alpha_matting_erode_size,
                post_process_mask=post_process_mask,
                bgcolor=bgcolor,
                detect_resolution=detect_resolution,
                upscale_method=upscale_method,
                **kwargs
            )
            ret_frames.append(result)
        
        # Send final frame completion
        if progress_callback is not None:
            progress_callback(total_frames, total_frames)
        
        return ret_frames

