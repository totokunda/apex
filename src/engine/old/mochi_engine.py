
from diffusers.video_processor import VideoProcessor

from src.engine.base_engine import BaseEngine
from src.engine.denoise.mochi_denoise import MochiDenoise

class MochiEngine(BaseEngine, MochiDenoise):
    def __init__(self, yaml_path: str, **kwargs):
        super().__init__(yaml_path, **kwargs)
        self.vae_spatial_scale_factor = (
            self.vae.config.get("scaling_factor", 8) if self.vae else 8
        )
        self.vae_temporal_scale_factor = 6  # Mochi specific

        self.video_processor = VideoProcessor(
            vae_scale_factor=self.vae_spatial_scale_factor
        )
        self.num_channels_latents = (
            self.transformer.config.in_channels if self.transformer else 12
        )

    

    
