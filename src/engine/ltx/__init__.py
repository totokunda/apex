from diffusers.video_processor import VideoProcessor
from src.utils.type import EnumType
from src.engine.base_engine import BaseEngine
from src.mixins import OffloadMixin
from src.denoise.ltx_denoise import LTXDenoise, DenoiseType
from src.mixins.loader_mixin import LoaderMixin


from .x2v import LTXX2VEngine


class ModelType(EnumType):
    X2V = "x2v" # Any  to video


class LTXEngine(BaseEngine, LoaderMixin, OffloadMixin, LTXDenoise):
    def __init__(
        self,
        yaml_path: str,
        model_type: ModelType = ModelType.X2V,
        **kwargs,
    ):
        super().__init__(yaml_path, **kwargs)

        self.model_type = model_type

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
            self.vae.config.latent_channels if self.vae is not None else 128
        )

        # Initialize the appropriate implementation engine
        self._init_implementation_engine()

    def _init_implementation_engine(self):
        """Initialize the specific implementation engine based on model type"""
        if self.model_type == ModelType.X2V:
            self.implementation_engine = LTXX2VEngine(self)
        else:
            raise ValueError(f"Model type {self.model_type} not supported")
