from src.utils.type import EnumType
from src.engine.base_engine import BaseEngine
from .t2i import ChromaT2IEngine
from diffusers.image_processor import VaeImageProcessor
from src.denoise.chroma_denoise import ChromaDenoise


class ModelType(EnumType):
    T2I = "t2i"  # text to image

class ChromaEngine(BaseEngine, ChromaDenoise):
    def __init__(self, yaml_path: str, model_type: ModelType = ModelType.T2I, **kwargs):

        super().__init__(yaml_path, model_type=model_type, **kwargs)

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1) if getattr(self, "vae", None) else 8
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
            self.implementation_engine = ChromaT2IEngine(self)
        else:
            raise ValueError(f"Invalid model type: {self.model_type}")

    def __str__(self):
        return f"ChromaEngine(config={self.config}, device={self.device}, model_type={self.model_type})"

    def __repr__(self):
        return self.__str__()
    
    def _render_step(self, latents, render_on_step_callback):
        impl = getattr(self, "implementation_engine", None)
        if impl is not None and hasattr(impl, "_render_step"):
            try:
                return impl._render_step(latents, render_on_step_callback)
            except Exception:
                pass
        # Fallback to BaseEngine behavior
        return super()._render_step(latents, render_on_step_callback)
