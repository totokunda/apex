from src.engine.base_engine import BaseEngine
import torch
from typing import Dict, Any, Callable, List, Union, Optional
from enum import Enum
from src.ui.nodes import UINode
from diffusers.video_processor import VideoProcessor
from src.engine.denoise import CogVideoDenoise, CogVideoDenoiseType

from .t2v import CogVideoT2VEngine
from .i2v import CogVideoI2VEngine
from .v2v import CogVideoV2VEngine
from .fun import CogVideoFunEngine


class ModelType(Enum):
    T2V = "t2v"  # text to video
    I2V = "i2v"  # image to video
    V2V = "v2v"  # video to video
    FUN = "fun"  # fun video


class CogVideoEngine(BaseEngine, CogVideoDenoise):
    def __init__(
        self,
        yaml_path: str,
        model_type: ModelType = ModelType.T2V,
        **kwargs,
    ):
        super().__init__(yaml_path, **kwargs)

        self.model_type = model_type

        if self.model_type == ModelType.I2V:
            self.denoise_type = CogVideoDenoiseType.I2V
        elif self.model_type == ModelType.V2V:
            self.denoise_type = CogVideoDenoiseType.V2V
        elif self.model_type == ModelType.FUN:
            self.denoise_type = CogVideoDenoiseType.FUN
        else:
            self.denoise_type = CogVideoDenoiseType.T2V


        self.vae_scale_factor_temporal = (
            getattr(self.vae, "config", {}).get("temporal_compression_ratio", None) or 4
            if getattr(self, "vae", None)
            else 4
        )
        self.vae_scale_factor_spatial = (
            2
            ** (
                len(
                    getattr(self.vae, "config", {}).get("block_out_channels", [1, 1, 1])
                )
                - 1
            )
            if getattr(self, "vae", None)
            else 8
        )
        self.vae_scaling_factor_image = (
            getattr(self.vae, "config", {}).get("scaling_factor", None) or 0.7
            if getattr(self, "vae", None)
            else 0.7
        )
        self.num_channels_latents = getattr(self.vae, "config", {}).get(
            "latent_channels", 16
        )
        self.video_processor = VideoProcessor(
            vae_scale_factor=self.vae_scale_factor_spatial
        )

        # Initialize the appropriate implementation engine
        self._init_implementation_engine()

    def _init_implementation_engine(self):
        """Initialize the specific implementation engine based on model type"""
        if self.model_type == ModelType.T2V:
            self.implementation_engine = CogVideoT2VEngine(self)
        elif self.model_type == ModelType.I2V:
            self.implementation_engine = CogVideoI2VEngine(self)
        elif self.model_type == ModelType.V2V:
            self.implementation_engine = CogVideoV2VEngine(self)
        elif self.model_type == ModelType.FUN:
            self.implementation_engine = CogVideoFunEngine(self)
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
        return f"CogVideoEngine(config={self.config}, device={self.device}, model_type={self.model_type})"

    def __repr__(self):
        return self.__str__()
