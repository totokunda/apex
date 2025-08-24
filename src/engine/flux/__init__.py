from src.utils.type import EnumType
from src.engine.base_engine import BaseEngine
from .t2i import FluxT2IEngine
from diffusers.image_processor import VaeImageProcessor
from src.denoise.flux_denoise import FluxDenoise


class ModelType(EnumType):
    T2I = "t2i"  # text to image


class FluxEngine(BaseEngine, FluxDenoise):
    def __init__(self, yaml_path: str, model_type: ModelType = ModelType.T2I, **kwargs):

        super().__init__(yaml_path, model_type=model_type, **kwargs)

        self.vae_scale_factor = (
            2 ** len(self.vae.temperal_downsample) if getattr(self, "vae", None) else 8
        )
        self.image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor * 2
        )
        self.num_channels_latents = (
            self.transformer.config.in_channels // 4 if self.transformer else 16
        )

        self._init_implementation_engine()

    def _init_implementation_engine(self):
        """Initialize the specific implementation engine based on model type"""
        if self.model_type == ModelType.T2I:
            self.implementation_engine = FluxT2IEngine(self)
        else:
            raise ValueError(f"Invalid model type: {self.model_type}")

    def __str__(self):
        return f"FluxEngine(config={self.config}, device={self.device}, model_type={self.model_type})"

    def __repr__(self):
        return self.__str__()
