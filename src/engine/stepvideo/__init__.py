from src.engine.base_engine import BaseEngine
import torch
from typing import List
from src.utils.type import EnumType
from src.ui.nodes import UINode
from diffusers.video_processor import VideoProcessor
from src.engine.denoise.stepvideo_denoise import StepVideoDenoise, DenoiseType
from .t2v import StepVideoT2VEngine
from .i2v import StepVideoI2VEngine


class ModelType(EnumType):
    T2V = "t2v"  # text to video
    I2V = "i2v"  # image to video


class StepVideoEngine(BaseEngine, StepVideoDenoise):
    def __init__(
        self,
        yaml_path: str,
        model_type: ModelType = ModelType.T2V,
        denoise_type: DenoiseType = DenoiseType.BASE,
        **kwargs,
    ):

        self.model_type = model_type
        self.denoise_type = denoise_type

        super().__init__(yaml_path, **kwargs)

        self.vae_scale_factor_temporal = (
            2 ** sum(self.vae.temporal_compression_ratio)
            if getattr(self.vae, "temporal_compression_ratio", None)
            else 8
        )

        self.vae_scale_factor_spatial = (
            2 ** len(self.vae.spatial_compression_ratio)
            if getattr(self.vae, "spatial_compression_ratio", None)
            else 16
        )

        self.num_channels_latents = getattr(self.vae, "config", {}).get(
            "z_channels", 16
        )
        self.video_processor = VideoProcessor(
            vae_scale_factor=self.vae_scale_factor_spatial
        )

        # Initialize the appropriate implementation engine
        self._init_implementation_engine()

    def _init_implementation_engine(self):
        """Initialize the specific implementation engine based on model type"""
        if self.model_type == ModelType.T2V:
            self.implementation_engine = StepVideoT2VEngine(self)
        elif self.model_type == ModelType.I2V:
            self.implementation_engine = StepVideoI2VEngine(self)
        else:
            raise ValueError(f"Invalid model type: {self.model_type}")

    @torch.inference_mode()
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
        return f"StepVideoEngine(config={self.config}, device={self.device}, model_type={self.model_type})"

    def __repr__(self):
        return self.__str__()
