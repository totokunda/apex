"""
PTLFlow optical flow preprocessor for video processing
Uses the ptlflow library for high-quality optical flow estimation
"""
import os
import warnings
from typing import List, Optional

import cv2
import numpy as np
import torch
from PIL import Image

from src.auxillary.util import HWC3, resize_image_with_pad
from src.types import InputImage, InputVideo, OutputImage, OutputVideo
from src.auxillary.base_preprocessor import BasePreprocessor
from src.mixins import ToMixin
from src.utils.defaults import get_torch_device, DEFAULT_PREPROCESSOR_SAVE_PATH


class PTLFlowDetector(ToMixin, BasePreprocessor):
    """
    PTLFlow optical flow detector for video processing.
    Uses state-of-the-art optical flow models from the ptlflow library.
    """
    
    def __init__(self, model, model_name: str = "raft"):
        super().__init__()
        self.model = model
        self.model_name = model_name
        self.device = get_torch_device()
        self.to_device(self.model, device=self.device)

    @classmethod
    def from_pretrained(cls, pretrained_model_or_path: str = "dpflow", ckpt_name: str = "sintel", **kwargs):
        """
        Load PTLFlow model from pretrained weights.
        
        Args:
            pretrained_model_or_path: Model name from ptlflow (e.g., "raft", "gma", "raft_small")
            ckpt_name: Checkpoint name - 'things', 'sintel', 'kitti', or path to custom checkpoint
            **kwargs: Additional arguments for model initialization
            
        Returns:
            PTLFlowDetector instance
        """
        try:
            # Import ptlflow
            import ptlflow
            from ptlflow.utils import flow_utils
            from ptlflow.utils.io_adapter import IOAdapter
        except ImportError:
            raise ImportError(
                "ptlflow is not installed. Please install it with: pip install ptlflow"
            )
        
        model_name = pretrained_model_or_path
        
        try:
            # Set PyTorch Hub cache directory to our preprocessor save path
            # PTLFlow uses ${TORCH_HOME}/hub/checkpoints/ for storing downloaded models
            torch_home = os.path.join(DEFAULT_PREPROCESSOR_SAVE_PATH, "torch")
            os.makedirs(torch_home, exist_ok=True)
            
            # Set TORCH_HOME environment variable so PyTorch Hub uses our cache location
            original_torch_home = os.environ.get('TORCH_HOME')
            os.environ['TORCH_HOME'] = torch_home
            
            try:
                # Load the model with checkpoint name (e.g., 'things', 'sintel', 'kitti')
                model = ptlflow.get_model(model_name, ckpt_path=ckpt_name)
                
                model.eval()
                model.to('cpu')  # Explicitly move to CPU
            finally:
                # Restore original TORCH_HOME if it was set
                if original_torch_home is not None:
                    os.environ['TORCH_HOME'] = original_torch_home
                elif 'TORCH_HOME' in os.environ:
                    del os.environ['TORCH_HOME']
            
        except Exception as e:
            raise ValueError(
                f"Failed to load ptlflow model '{model_name}'. "
                f"Error: {str(e)}. "
                f"Available models: raft, raft_small, gma, craft, flowformer, etc."
            )
        
        return cls(model, model_name)
    
    def process_image(self, input_image: InputImage, **kwargs) -> OutputImage:
        """PTLFlow does not support single image processing"""
        raise NotImplementedError(
            "PTLFlowDetector only processes videos. Use process_video() or call with a video input."
        )
    
    def process(self, frame1, frame2=None, detect_resolution=512, upscale_method="INTER_CUBIC", output_type: str = "vis", **kwargs):
        """
        Process two consecutive frames to compute optical flow.
        
        Args:
            frame1: First frame (current frame)
            frame2: Second frame (next frame)
            detect_resolution: Resolution for processing (0 = original size)
            upscale_method: Method for resizing ("INTER_CUBIC", "INTER_LINEAR", etc.)
            output_type: "vis" for visualization, "flow" for raw flow data
            **kwargs: Additional processing parameters
            
        Returns:
            Flow visualization image or raw flow array
        """
        if frame2 is None:
            raise ValueError("PTLFlowDetector.process() requires two frames (frame1 and frame2)")
        
        # Import ptlflow utilities
        from ptlflow.utils import flow_utils
        from ptlflow.utils.io_adapter import IOAdapter
        
        # Convert to numpy arrays if needed
        if isinstance(frame1, Image.Image):
            frame1 = np.array(frame1)
        if isinstance(frame2, Image.Image):
            frame2 = np.array(frame2)
        
        # Ensure RGB format
        if frame1.ndim == 2:
            frame1 = cv2.cvtColor(frame1, cv2.COLOR_GRAY2RGB)
        if frame2.ndim == 2:
            frame2 = cv2.cvtColor(frame2, cv2.COLOR_GRAY2RGB)
        
        # Resize frames if detect_resolution is specified
        if detect_resolution > 0:
            frame1_resized, remove_pad1 = resize_image_with_pad(frame1, detect_resolution, upscale_method)
            frame2_resized, remove_pad2 = resize_image_with_pad(frame2, detect_resolution, upscale_method)
        else:
            frame1_resized = frame1
            frame2_resized = frame2
            remove_pad1 = lambda x: x
            remove_pad2 = lambda x: x
        
        # PTLFlow expects BGR format (OpenCV format)
        frame1_bgr = cv2.cvtColor(frame1_resized, cv2.COLOR_RGB2BGR)
        frame2_bgr = cv2.cvtColor(frame2_resized, cv2.COLOR_RGB2BGR)
        
        images = [frame1_bgr, frame2_bgr]
        
        # Use IOAdapter to prepare inputs properly
        io_adapter = IOAdapter(self.model, frame1_bgr.shape[:2])
        
        with torch.no_grad():
            # Prepare inputs - returns dict with 'images' key
            # Tensor shape: (1, 2, 3, H, W) - BNCHW
            inputs = io_adapter.prepare_inputs(images)
            
            # Move inputs to device
            inputs['images'] = inputs['images'].to(self.device)
            
            # Forward through model
            predictions = self.model(inputs)
            
            # Extract flow from predictions
            # flows shape: (1, 1, 2, H, W) - BNCHW
            flows = predictions['flows']
            
            # Return based on output_type
            if output_type == "flow":
                # Return as numpy array [H, W, 2]
                flow_np = flows[0, 0].permute(1, 2, 0).cpu().numpy()
                # Remove padding to restore original size
                flow_np = remove_pad1(flow_np)
                return flow_np
            else:  # "vis"
                # Create RGB representation using ptlflow's utility
                flow_rgb = flow_utils.flow_to_rgb(flows)
                # Convert from (1, 1, 3, H, W) to (H, W, 3)
                flow_rgb = flow_rgb[0, 0].permute(1, 2, 0)
                flow_rgb_npy = flow_rgb.detach().cpu().numpy()
                
                # flow_rgb_npy is already in RGB format, convert to uint8
                flow_rgb_uint8 = (flow_rgb_npy * 255).astype(np.uint8)
                
                # Remove padding to restore original size
                flow_rgb_uint8 = remove_pad1(flow_rgb_uint8)
                
                return Image.fromarray(flow_rgb_uint8)
    
    def process_video(self, input_video: InputVideo, output_type: str = "vis", **kwargs) -> OutputVideo:
        """
        Process video to compute optical flow between consecutive frames.
        
        Args:
            input_video: Input video as list of frames or video path
            output_type: "vis" for visualization, "flow" for raw flow data
            **kwargs: Additional processing parameters (including progress_callback)
            
        Returns:
            List of flow visualization images (or flow arrays if output_type="flow")
        """
        from tqdm import tqdm
        
        frames = self._load_video(input_video)
        
        if len(frames) < 2:
            raise ValueError(f"Video must have at least 2 frames for optical flow, got {len(frames)}")
        
        ret_frames = []
        
        # Duplicate last frame to ensure all frames get processed
        frames_extended = frames + [frames[-1]]
        
        # Get progress callback if provided
        progress_callback = kwargs.get("progress_callback", None)
        total_frames = len(frames)
        
        # Process each pair of consecutive frames
        for frame_idx in tqdm(range(total_frames), desc="Processing optical flow"):
            # Update progress
            if progress_callback is not None:
                progress_callback(frame_idx + 1, total_frames)
            
            # Get consecutive frames
            frame1 = frames_extended[frame_idx]
            frame2 = frames_extended[frame_idx + 1]
            
            # Process the frame pair
            result = self.process(frame1, frame2, output_type=output_type, **kwargs)
            ret_frames.append(result)
        
        # Send final frame completion
        if progress_callback is not None:
            progress_callback(total_frames, total_frames)
        
        return ret_frames

