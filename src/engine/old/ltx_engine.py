import torch
from diffusers.video_processor import VideoProcessor
from enum import Enum, auto
from typing import List

from src.engine.wan_engine import BaseEngine, OffloadMixin
from src.engine.denoise.ltx_denoise import LTXDenoise, DenoiseType
from src.ui.nodes import UINode
from typing import Dict, Any, Callable
import math
from PIL import Image
import numpy as np
from typing import Union
from typing import Optional, Tuple
from diffusers.models.autoencoders import AutoencoderKL
from diffusers.utils.torch_utils import randn_tensor
import inspect
from src.mixins.loader_mixin import LoaderMixin


class ModelType(Enum):
    T2V = "t2v"  # text to video
    I2V = "i2v"  # image to video
    CONTROL = "control"




class LTXEngine(BaseEngine, OffloadMixin, LTXDenoise):
    def __init__(
        self,
        yaml_path: str,
        model_type: ModelType = ModelType.T2V,
        **kwargs,
    ):
        super().__init__(yaml_path, **kwargs)

        self.model_type = model_type
        if self.model_type == ModelType.CONTROL:
            self.denoise_type = DenoiseType.CONDITION
        elif self.model_type == ModelType.T2V:
            self.denoise_type = DenoiseType.T2V
        elif self.model_type == ModelType.I2V:
            self.denoise_type = DenoiseType.I2V
        else:
            raise ValueError(f"Model type {self.model_type} not supported")

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
            self.vae.config.get("latent_channels", 128) if self.vae is not None else 128
        )

    def run(self, *args, input_nodes: List[UINode] = None, **kwargs):
        default_kwargs = self._get_default_kwargs("run")
        preprocessed_kwargs = self._preprocess_kwargs(input_nodes, **kwargs)
        final_kwargs = {**default_kwargs, **preprocessed_kwargs}

        if self.model_type == ModelType.T2V:
            return self.t2v_run(**final_kwargs)
        elif self.model_type == ModelType.I2V:
            return self.i2v_run(**final_kwargs)
        elif self.model_type == ModelType.CONTROL:
            return self.control_run(**final_kwargs)
        else:
            raise ValueError(f"Model type {self.model_type} not supported")



    

    

    


if __name__ == "__main__":
    from diffusers.utils import export_to_video
    from PIL import Image

    engine = LTXEngine(
        yaml_path="manifest/ltx_x2v_13b.yml",
        model_type=ModelType.T2V,
        save_path="/tmp/apex_models",  # Change this to your desired save path
        components_to_load=["transformer", "vae", "text_encoder", "scheduler"],
        postprocessors_to_load=["latent_upscaler"],
    )

    prompt = "A majestic lion basking in the golden hour sun"
    negative_prompt = "worst quality, inconsistent motion, blurry, jittery, distorted"

    height = 480
    width = 832
    print(f"height: {height}, width: {width}")

    video = engine.run(
        height=height,
        width=width,
        prompt=prompt,
        negative_prompt=negative_prompt,
        use_cfg_guidance=True,
        duration="25f",
        num_videos=1,
        guidance_scale=3.0,
        num_inference_steps=30,
        generator=torch.Generator(device="cuda").manual_seed(42),
        postprocessor_kwargs={"adain_factor": 0.5},
    )

    export_to_video(video[0], "t2v_ltx2v_upscaled.mp4", fps=24, quality=8)
