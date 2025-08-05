from src.engine.base_engine import BaseEngine
import torch
from typing import List
from src.utils.type_utils import EnumType
from src.ui.nodes import UINode
from diffusers.video_processor import VideoProcessor
from src.engine.denoise import MagiDenoise, MagiDenoiseType

from .t2v import MagiT2VEngine
from .i2v import MagiI2VEngine
from .v2v import MagiV2VEngine


class ModelType(EnumType):
    T2V = "t2v"  # text to video
    I2V = "i2v"  # image to video
    V2V = "v2v"  # video to video


class MagiEngine(BaseEngine, MagiDenoise):
    def __init__(
        self,
        yaml_path: str,
        model_type: ModelType = ModelType.T2V,
        denoise_type: MagiDenoiseType = MagiDenoiseType.BASE,
        **kwargs,
    ):

        self.model_type = model_type
        self.denoise_type = denoise_type

        super().__init__(yaml_path, **kwargs)

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

        # Initialize the appropriate implementation engine
        self._init_implementation_engine()

    def _init_implementation_engine(self):
        """Initialize the specific implementation engine based on model type"""
        if self.model_type == ModelType.T2V:
            self.implementation_engine = MagiT2VEngine(self)
        elif self.model_type == ModelType.I2V:
            self.implementation_engine = MagiI2VEngine(self)
        elif self.model_type == ModelType.V2V:
            self.implementation_engine = MagiV2VEngine(self)
        else:
            raise ValueError(f"Invalid model type: {self.model_type}")

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

        return self.implementation_engine.run(**final_kwargs)

    def __str__(self):
        return f"MagiEngine(config={self.config}, device={self.device}, model_type={self.model_type})"

    def __repr__(self):
        return self.__str__()
