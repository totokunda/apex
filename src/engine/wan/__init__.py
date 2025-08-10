from src.engine.base_engine import BaseEngine
import torch
from typing import List
from src.ui.nodes import UINode
from diffusers.video_processor import VideoProcessor
from src.engine.denoise.wan_denoise import WanDenoise, DenoiseType

from .t2v import WanT2VEngine
from .i2v import WanI2VEngine
from .vace import WanVaceEngine
from .fflf import WanFFLFEngine
from .causal import WanCausalEngine
from .fun import WanFunEngine
from .control import WanControlEngine
from .inp import WanInpEngine
from .phantom import WanPhantomEngine
from .apex_framepack import WanApexFramepackEngine
from .multitalk import WanMultitalkEngine
from src.utils.type import EnumType


class ModelType(EnumType):
    VACE = "vace"  # vace
    T2V = "t2v"  # text to video
    I2V = "i2v"  # image to video
    FFLF = "fflf"  # first frame last frame
    CAUSAL = "causal"  # causal
    FUN = "fun"  # fun (combined)
    CONTROL = "control"  # control (camera poses/video control)
    INP = "inp"  # inpainting (video + mask)
    PHANTOM = "phantom"  # phantom (subject reference images)
    APEX_FRAMEPACK = "apex_framepack"  # apex framepack
    MULTITALK = "multitalk"  # multitalk audio-driven video


class WanEngine(BaseEngine, WanDenoise):
    def __init__(
        self,
        yaml_path: str,
        model_type: ModelType = ModelType.T2V,
        denoise_type: DenoiseType = DenoiseType.BASE,
        **kwargs,
    ):
        self.model_type = model_type
        self.denoise_type = denoise_type

        super().__init__(yaml_path, **kwargs)

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
            vae_scale_factor=kwargs.get(
                "vae_scale_factor", self.vae_scale_factor_spatial
            )
        )

        # Initialize the appropriate implementation engine
        self._init_implementation_engine()

    def _init_implementation_engine(self):
        """Initialize the specific implementation engine based on model type"""
        if self.model_type == ModelType.T2V:
            self.implementation_engine = WanT2VEngine(self)
        elif self.model_type == ModelType.VACE:
            self.implementation_engine = WanVaceEngine(self)
        elif self.model_type == ModelType.I2V:
            self.implementation_engine = WanI2VEngine(self)
        elif self.model_type == ModelType.FFLF:
            self.implementation_engine = WanFFLFEngine(self)
        elif self.model_type == ModelType.CAUSAL:
            self.implementation_engine = WanCausalEngine(self)
        elif self.model_type == ModelType.FUN:
            self.implementation_engine = WanFunEngine(self)
        elif self.model_type == ModelType.CONTROL:
            self.implementation_engine = WanControlEngine(self)
        elif self.model_type == ModelType.INP:
            self.implementation_engine = WanInpEngine(self)
        elif self.model_type == ModelType.PHANTOM:
            self.implementation_engine = WanPhantomEngine(self)
        elif self.model_type == ModelType.APEX_FRAMEPACK:
            self.implementation_engine = WanApexFramepackEngine(self)
        elif self.model_type == ModelType.MULTITALK:
            self.implementation_engine = WanMultitalkEngine(self)
        else:
            raise ValueError(f"Invalid model type: {self.model_type}")

    @torch.inference_mode()
    def run(
        self,
        input_nodes: List[UINode] = None,
        **kwargs,
    ):
        default_kwargs = self._get_default_kwargs("run")
        preprocessed_kwargs = self._preprocess_kwargs(input_nodes, **kwargs)
        final_kwargs = {**default_kwargs, **preprocessed_kwargs}

        return self.implementation_engine.run(**final_kwargs)

    def __str__(self):
        return f"WanEngine(config={self.config}, device={self.device}, model_type={self.model_type})"

    def __repr__(self):
        return self.__str__()
