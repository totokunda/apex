from diffusers.video_processor import VideoProcessor
from typing import List
from src.utils.type import EnumType
from src.engine.base_engine import BaseEngine
from src.mixins import OffloadMixin
from src.engine.denoise.ltx_denoise import LTXDenoise, DenoiseType
from src.ui.nodes import UINode
from typing import Dict, Any, Callable
import math
from PIL import Image
from src.mixins.loader_mixin import LoaderMixin

from .t2v import LTXT2VEngine
from .i2v import LTXI2VEngine
from .control import LTXControlEngine


class ModelType(EnumType):
    T2V = "t2v"  # text to video
    I2V = "i2v"  # image to video
    CONTROL = "control"


class LTXEngine(BaseEngine, LoaderMixin, OffloadMixin, LTXDenoise):
    def __init__(
        self,
        yaml_path: str,
        model_type: ModelType = ModelType.T2V,
        **kwargs,
    ):
        super().__init__(yaml_path, **kwargs)

        self.model_type = model_type

        if self.model_type == ModelType.CONTROL:
            self.denoise_type = DenoiseType.CONDITION
        elif self.model_type == ModelType.T2V:
            self.denoise_type = DenoiseType.T2V
        elif self.model_type == ModelType.I2V:
            self.denoise_type = DenoiseType.I2V
        else:
            raise ValueError(f"Model type {self.model_type} not supported")

        self.vae_scale_factor_spatial = (
            self.vae.spatial_compression_ratio
            if getattr(self, "vae", None) is not None
            else 32
        )

        self.vae_scale_factor_temporal = (
            self.vae.temporal_compression_ratio
            if getattr(self, "vae", None) is not None
            else 8
        )

        self.transformer_spatial_patch_size = (
            self.transformer.config.patch_size
            if getattr(self, "transformer", None) is not None
            else 1
        )

        self.transformer_temporal_patch_size = (
            self.transformer.config.patch_size_t
            if getattr(self, "transformer") is not None
            else 1
        )

        self.video_processor = VideoProcessor(
            vae_scale_factor=self.vae_scale_factor_spatial
        )

        self.num_channels_latents: int = (
            self.vae.config.get("latent_channels", 128) if self.vae is not None else 128
        )

        # Initialize the appropriate implementation engine
        self._init_implementation_engine()

    def _init_implementation_engine(self):
        """Initialize the specific implementation engine based on model type"""
        if self.model_type == ModelType.T2V:
            self.implementation_engine = LTXT2VEngine(self)
        elif self.model_type == ModelType.I2V:
            self.implementation_engine = LTXI2VEngine(self)
        elif self.model_type == ModelType.CONTROL:
            self.implementation_engine = LTXControlEngine(self)
        else:
            raise ValueError(f"Model type {self.model_type} not supported")
