from diffusers.video_processor import VideoProcessor
from typing import List
from src.utils.type import EnumType
from src.engine.base_engine import BaseEngine
from src.mixins import OffloadMixin
from src.engine.denoise.cosmos2_denoise import Cosmos2Denoise, DenoiseType
from src.ui.nodes import UINode
from typing import Dict, Any, Callable
import math
from PIL import Image
from src.mixins.loader_mixin import LoaderMixin

from .i2v import Cosmos2I2VEngine
from .v2v import Cosmos2V2VEngine

class ModelType(EnumType):
    I2V = "i2v"  # image to video
    V2V = "v2v"  # video to video


class Cosmos2Engine(BaseEngine, LoaderMixin, OffloadMixin, Cosmos2Denoise):
    def __init__(
        self,
        yaml_path: str,
        model_type: ModelType = ModelType.I2V,
        **kwargs,
    ):
        super().__init__(yaml_path, **kwargs)

        self.model_type = model_type

        if self.model_type == ModelType.I2V or self.model_type == ModelType.V2V:
            self.denoise_type = DenoiseType.BASE
        else:
            raise ValueError(f"Model type {self.model_type} not supported")

        self.vae_scale_factor_temporal = (
            2 ** sum(self.vae.temperal_downsample)
            if getattr(self.vae, "temperal_downsample", None)
            else 4
        )

        self.vae_scale_factor_spatial = (
            2 ** len(self.vae.temperal_downsample)
            if getattr(self.vae, "temperal_downsample", None)
            else 8
        )

        self.num_channels_latents = getattr(self.vae, "config", {}).get("z_dim", 16)

        self.video_processor = VideoProcessor(
            vae_scale_factor=self.vae_scale_factor_spatial
        )

        # Initialize the appropriate implementation engine
        self._init_implementation_engine()

    def _init_implementation_engine(self):
        """Initialize the specific implementation engine based on model type"""
        if self.model_type == ModelType.I2V:
            self.implementation_engine = Cosmos2I2VEngine(self)
        elif self.model_type == ModelType.V2V:
            self.implementation_engine = Cosmos2V2VEngine(self)
        else:
            raise ValueError(f"Model type {self.model_type} not supported")

    
