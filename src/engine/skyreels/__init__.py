from src.utils.type_utils import EnumType
from src.engine.denoise.wan_denoise import DenoiseType
import torch
from typing import List, Union, Optional
from src.ui.nodes import UINode
from PIL import Image
from typing import Dict, Any, Callable
import numpy as np

from .t2v import SkyReelsT2VEngine
from .i2v import SkyReelsI2VEngine
from .df import SkyReelsDFEngine
from src.engine.wan import WanEngine


class ModelType(EnumType):
    T2V = "t2v"  # text to video
    I2V = "i2v"  # image to video
    DF = "df"  # diffusion forcing


class SkyReelsEngine(WanEngine):
    def __init__(self, yaml_path: str, model_type: ModelType = ModelType.T2V, **kwargs):
        if model_type == ModelType.DF:
            denoise_type = DenoiseType.DIFFUSION_FORCING
        else:
            denoise_type = DenoiseType.BASE
        super().__init__(
            yaml_path, model_type=model_type, **kwargs
        )
        self.denoise_type = denoise_type

        # Initialize the appropriate implementation engine
        self._init_implementation_engine()

    def _init_implementation_engine(self):
        """Initialize the specific implementation engine based on model type"""
        if self.model_type == ModelType.T2V:
            self.implementation_engine = SkyReelsT2VEngine(self)
        elif self.model_type == ModelType.I2V:
            self.implementation_engine = SkyReelsI2VEngine(self)
        elif self.model_type == ModelType.DF:
            self.implementation_engine = SkyReelsDFEngine(self)
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
        return f"SkyReelsEngine(config={self.config}, device={self.device}, model_type={self.model_type})"

    def __repr__(self):
        return self.__str__()
