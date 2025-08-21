from src.engine.base_engine import BaseEngine
from src.utils.type import EnumType
from diffusers.video_processor import VideoProcessor
from src.denoise.hunyuan_denoise import DenoiseType as HunyuanDenoiseType, HunyuanDenoise

from .t2v import HunyuanT2VEngine
from .i2v import HunyuanI2VEngine
from .framepack import HunyuanFramepackEngine
from .avatar import HunyuanAvatarEngine


class ModelType(EnumType):
    T2V = "t2v"  # text to video
    I2V = "i2v"  # image to video
    FRAMEPACK = "framepack"  # framepack
    AVATAR = "avatar"  # avatar

class HunyuanEngine(BaseEngine, HunyuanDenoise):
    def __init__(
        self,
        yaml_path: str,
        model_type: ModelType = ModelType.T2V,
        denoise_type: HunyuanDenoiseType = HunyuanDenoiseType.BASE,
        **kwargs,
    ):
        super().__init__(yaml_path, **kwargs)

        self.model_type = model_type
        self.denoise_type = denoise_type

        if model_type == ModelType.AVATAR:
            self.denoise_type = HunyuanDenoiseType.AVATAR

        self.vae_scale_factor_temporal = (
            getattr(self.vae, "temporal_compression_ratio", None) or 4
            if getattr(self, "vae", None)
            else 4
        )
        self.vae_scale_factor_spatial = (
            getattr(self.vae, "spatial_compression_ratio", None) or 8
            if getattr(self, "vae", None)
            else 8
        )
        self.num_channels_latents = getattr(self.vae, "config", {}).get(
            "latent_channels", 16
        )
        self.video_processor = VideoProcessor(
            vae_scale_factor=self.vae_scale_factor_spatial
        )
        self.llama_text_encoder = None

        # Initialize the appropriate implementation engine
        self._init_implementation_engine()

    def _init_implementation_engine(self):
        """Initialize the specific implementation engine based on model type"""
        if self.model_type == ModelType.T2V:
            self.implementation_engine = HunyuanT2VEngine(self)
        elif self.model_type == ModelType.I2V:
            self.implementation_engine = HunyuanI2VEngine(self)
        elif self.model_type == ModelType.FRAMEPACK:
            self.implementation_engine = HunyuanFramepackEngine(self)
        elif self.model_type == ModelType.AVATAR:
            self.implementation_engine = HunyuanAvatarEngine(self)
        else:
            raise ValueError(f"Invalid model type: {self.model_type}")

    def __str__(self):
        return f"HunyuanEngine(config={self.config}, device={self.device}, model_type={self.model_type})"

    def __repr__(self):
        return self.__str__()
