from src.utils.type import EnumType
from src.engine.base_engine import BaseEngine
from .t2i import HunyuanImageT2IEngine
from diffusers.image_processor import VaeImageProcessor
from src.denoise.hunyuanimage_denoise import HunyuanImageDenoise

class ModelType(EnumType):
    T2I = "t2i"  # text to image

class HunyuanImageEngine(BaseEngine, HunyuanImageDenoise):
    def __init__(self, yaml_path: str, model_type: ModelType = ModelType.T2I, **kwargs):

        super().__init__(yaml_path, model_type=model_type, **kwargs)

        self.vae_scale_factor = self.vae.config.spatial_compression_ratio if getattr(self, "vae", None) else 32
        
        self.image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor
        )

        self._init_implementation_engine()

    def _init_implementation_engine(self):
        """Initialize the specific implementation engine based on model type"""
        if self.model_type == ModelType.T2I:
            self.implementation_engine = HunyuanImageT2IEngine(self)
        else:
            raise ValueError(f"Invalid model type: {self.model_type}")

    def __str__(self):
        return f"HunyuanImageEngine(config={self.config}, device={self.device}, model_type={self.model_type})"

    def __repr__(self):
        return self.__str__()

    # Ensure denoise loop uses implementation-specific render step
    def _render_step(self, latents, render_on_step_callback):
        impl = getattr(self, "implementation_engine", None)
        if impl is not None and hasattr(impl, "_render_step"):
            try:
                return impl._render_step(latents, render_on_step_callback)
            except Exception:
                pass
        # Fallback to BaseEngine behavior
        return super()._render_step(latents, render_on_step_callback)
