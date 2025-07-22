from src.engine.base_engine import BaseEngine
import torch
from typing import Dict, Any, Callable, List, Union, Optional, Tuple
from enum import Enum
from src.ui.nodes import UINode
from diffusers.video_processor import VideoProcessor
from PIL import Image
import numpy as np
from src.engine.denoise import MagiDenoise, MagiDenoiseType
import math


class ModelType(Enum):
    T2V = "t2v"  # text to video
    I2V = "i2v"  # image to video
    V2V = "v2v"  # video to video


class MagiEngine(BaseEngine, MagiDenoise):
    def __init__(
        self,
        yaml_path: str,
        model_type: ModelType = ModelType.T2V,
        denoise_type: MagiDenoiseType = MagiDenoiseType.T2V,
        **kwargs,
    ):
        super().__init__(yaml_path, **kwargs)

        self.model_type = model_type
        self.denoise_type = denoise_type

        # Set up VAE scale factors based on MAGI VAE configuration
        self.vae_scale_factor_temporal = (
            getattr(self.vae, "temporal_compression_ratio", None)
            or getattr(self.vae, "patch_length", None)
            or 4
            if getattr(self, "vae", None)
            else 4
        )
        self.vae_scale_factor_spatial = (
            getattr(self.vae, "spatial_compression_ratio", None)
            or getattr(self.vae, "patch_size", None)
            or 8
            if getattr(self, "vae", None)
            else 8
        )

        # MAGI uses different channel configurations
        self.num_channels_latents = getattr(self.vae, "config", {}).get(
            "z_chans", 4  # MAGI default
        )

        self.video_processor = VideoProcessor(
            vae_scale_factor=self.vae_scale_factor_spatial
        )

    @torch.inference_mode()
    def run(
        self,
        input_nodes: List[UINode] = None,
        **kwargs,
    ):
        """Main run method that routes to appropriate generation function"""
        default_kwargs = self._get_default_kwargs("run")
        preprocessed_kwargs = self._preprocess_kwargs(input_nodes, **kwargs)
        final_kwargs = {**default_kwargs, **preprocessed_kwargs}

        if self.model_type == ModelType.T2V:
            return self.t2v_run(**final_kwargs)
        elif self.model_type == ModelType.I2V:
            return self.i2v_run(**final_kwargs)
        elif self.model_type == ModelType.V2V:
            return self.v2v_run(**final_kwargs)
        else:
            raise ValueError(f"Invalid model type: {self.model_type}")

    def _load_video(self, video):
        """Load video from various input formats"""
        # Placeholder - implement based on your video loading utilities
        if isinstance(video, str):
            # Load from file path
            pass
        elif isinstance(video, list):
            # Load from list of images or paths
            pass
        elif isinstance(video, (np.ndarray, torch.Tensor)):
            # Already loaded
            pass
        # Add more loading logic as needed
        return video

    def _aspect_ratio_resize_video(self, video, max_area):
        """Resize video maintaining aspect ratio"""
        # Placeholder - implement based on your video processing utilities
        height, width = video.shape[-2:]  # Assuming THWC or similar format
        return video, height, width

    def __str__(self):
        return f"MagiEngine(config={self.config}, device={self.device}, model_type={self.model_type})"

    def __repr__(self):
        return self.__str__()


if __name__ == "__main__":
    from diffusers.utils import export_to_video

    # Example usage for text-to-video
    engine = MagiEngine(
        yaml_path="manifest/magi_t2v.yml",  # You'll need to create this
        model_type=ModelType.T2V,
        save_path="./apex-models",
        components_to_load=["transformer", "text_encoder", "vae", "scheduler"],
        component_dtypes={"vae": torch.float16, "transformer": torch.float16},
    )

    prompt = "A serene sunset over a calm lake with gentle ripples"
    video = engine.run(
        height=512,
        width=512,
        duration=5,
        prompt=prompt,
        num_inference_steps=50,
        guidance_scale=6.0,
        fps=24,
        seed=42,
    )

    export_to_video(video[0], "magi_t2v_output.mp4", fps=24)
