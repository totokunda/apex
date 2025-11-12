from src.utils.type import EnumType
from src.engine.base_engine import BaseEngine
from .t2i import FluxT2IEngine
from .kontext import FluxKontextEngine
from .fill import FluxFillEngine
from .control import FluxControlEngine
from diffusers.image_processor import VaeImageProcessor
from src.denoise.flux_denoise import FluxDenoise
from .dreamomni2 import DreamOmni2Engine


class ModelType(EnumType):
    T2I = "t2i"  # text to image
    KONTEXT = "kontext"  # kontext
    FILL = "fill"  # fill
    CONTROL = "control"  # control
    DREAMOMNI2 = "dreamomni2"  # dreamomni2
    
class FluxEngine(BaseEngine, FluxDenoise):
    def __init__(self, yaml_path: str, model_type: ModelType = ModelType.T2I, **kwargs):

        super().__init__(yaml_path, model_type=model_type, **kwargs)

        self.vae_scale_factor = (
            2 ** len(self.vae.temperal_downsample) if getattr(self, "vae", None) else 8
        )
        self.image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor * 2
        )
        if self.model_type == ModelType.CONTROL:
            self.num_channels_latents = (
                self.transformer.config.in_channels // 8 if self.transformer else 16
            )
        else:
            self.num_channels_latents = (
                self.transformer.config.in_channels // 4 if self.transformer else 16
            )
        
        self.default_sample_size = 128

        self._init_implementation_engine()

    def _init_implementation_engine(self):
        """Initialize the specific implementation engine based on model type"""
        if self.model_type == ModelType.T2I:
            self.implementation_engine = FluxT2IEngine(self)
        elif self.model_type == ModelType.KONTEXT:
            self.implementation_engine = FluxKontextEngine(self)
        elif self.model_type == ModelType.FILL:
            self.implementation_engine = FluxFillEngine(self)
        elif self.model_type == ModelType.CONTROL:
            self.implementation_engine = FluxControlEngine(self)
        elif self.model_type == ModelType.DREAMOMNI2:
            self.implementation_engine = DreamOmni2Engine(self)
        else:
            raise ValueError(f"Invalid model type: {self.model_type}")

    def __str__(self):
        return f"FluxEngine(config={self.config}, device={self.device}, model_type={self.model_type})"

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
