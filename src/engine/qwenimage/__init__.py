from src.utils.type import EnumType
from src.engine.base_engine import BaseEngine
from .t2i import QwenImageT2IEngine
from .edit import QwenImageEditEngine
from .controlnet import QwenImageControlNetEngine
from diffusers.image_processor import VaeImageProcessor
from src.denoise.qwenimage_denoise import QwenImageDenoise
from .edit_plus import QwenImageEditPlusEngine

class ModelType(EnumType):
    T2I = "t2i"  # text to image
    EDIT = "edit"  # edit
    CONTROLNET = "controlnet"  # controlnet
    EDIT_PLUS = "edit_plus"  # edit plus

class QwenImageEngine(BaseEngine, QwenImageDenoise):
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
            self.implementation_engine = QwenImageT2IEngine(self)
        elif self.model_type == ModelType.EDIT:
            self.implementation_engine = QwenImageEditEngine(self)
        elif self.model_type == ModelType.CONTROLNET:
            self.implementation_engine = QwenImageControlNetEngine(self)
        elif self.model_type == ModelType.EDIT_PLUS:
            self.implementation_engine = QwenImageEditPlusEngine(self)
        else:
            raise ValueError(f"Invalid model type: {self.model_type}")

    def __str__(self):
        return f"QwenImageEngine(config={self.config}, device={self.device}, model_type={self.model_type})"

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
