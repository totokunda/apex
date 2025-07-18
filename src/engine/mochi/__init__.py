from typing import List
from enum import Enum
from diffusers.video_processor import VideoProcessor
from src.engine.denoise.mochi_denoise import MochiDenoise
from src.ui.nodes import UINode


from src.engine.base_engine import BaseEngine
from .t2v import MochiT2VEngine

class ModelType(Enum):
    T2V = "t2v"  # text to video

class MochiEngine(BaseEngine, MochiDenoise):
    def __init__(self, yaml_path: str, model_type: ModelType = ModelType.T2V, **kwargs):
        super().__init__(yaml_path, model_type=ModelType.T2V, **kwargs)
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
        
        self.model_type = model_type
        # Initialize the T2V implementation engine
        self.implementation_engine = MochiT2VEngine(self)

    def run(
        self,
        input_nodes: List[UINode] = None,
        **kwargs,
    ):
        default_kwargs = self._get_default_kwargs("run")
        preprocessed_kwargs = self._preprocess_kwargs(input_nodes, **kwargs)
        final_kwargs = {**default_kwargs, **preprocessed_kwargs}
        
        return self.implementation_engine.run(**final_kwargs)

    def __str__(self):
        return f"MochiEngine(config={self.config}, device={self.device})"

    def __repr__(self):
        return self.__str__() 