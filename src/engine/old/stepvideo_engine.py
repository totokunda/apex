from src.engine.base_engine import BaseEngine
import torch
from typing import Dict, Any, Callable
from enum import Enum
from src.ui.nodes import UINode
from typing import List
from diffusers.video_processor import VideoProcessor
import math
from PIL import Image
import numpy as np
from typing import Union
from src.mixins import OffloadMixin
import torchvision.transforms.functional as TF
from src.engine.denoise.stepvideo_denoise import StepVideoDenoise, DenoiseType
import torch.nn.functional as F


class ModelType(Enum):
    T2V = "t2v"  # text to video


class StepVideoEngine(BaseEngine, StepVideoDenoise):
    def __init__(
        self,
        yaml_path: str,
        model_type: ModelType = ModelType.T2V,
        denoise_type: DenoiseType = DenoiseType.BASE,
        **kwargs,
    ):
        super().__init__(yaml_path, **kwargs)

        self.model_type = model_type
        self.denoise_type = denoise_type

        self.vae_scale_factor_temporal = (
            2 ** sum(self.vae.temporal_compression_ratio)
            if getattr(self.vae, "temporal_compression_ratio", None)
            else 4
        )

        self.vae_scale_factor_spatial = (
            2 ** len(self.vae.spatial_compression_ratio)
            if getattr(self.vae, "spatial_compression_ratio", None)
            else 8
        )
        self.num_channels_latents = getattr(self.vae, "config", {}).get("z_dim", 16)
        self.video_processor = VideoProcessor(
            vae_scale_factor=self.vae_scale_factor_spatial
        )

    @torch.inference_mode()
    def run(
        self,
        input_nodes: List[UINode] = None,
        **kwargs,
    ):
        default_kwargs = self._get_default_kwargs("run")
        preprocessed_kwargs = self._preprocess_kwargs(input_nodes, **kwargs)
        final_kwargs = {**default_kwargs, **preprocessed_kwargs}

        if self.model_type == ModelType.T2V:
            return self.t2v_run(**final_kwargs)
        else:
            raise ValueError(f"Invalid model type: {self.model_type}")

    def __str__(self):
        return f"StepVideoEngine(config={self.config}, device={self.device}, model_type={self.model_type})"

    def __repr__(self):
        return self.__str__()


if __name__ == "__main__":
    from diffusers.utils import export_to_video
    from PIL import Image

    engine = StepVideoEngine(
        yaml_path="manifest/stepvideo_t2.yml",
        model_type=ModelType.T2V,
        save_path="./apex-models",  # Change this to your desired save path,  # Change this to your desired save path
        components_to_load=["transformer"],
        component_dtypes={"vae": torch.float16},
    )

    prompt = "A stylish woman walks down a Tokyo street filled with warm glowing neon and animated city signage. She wears a black leather jacket, a long red dress, and black boots, and carries a black purse. She wears sunglasses and red lipstick. She walks confidently and casually. The street is damp and reflective, creating a mirror effect of the colorful lights. Many pedestrians walk about."
    negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
    # negative_prompt = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"

    height = 480
    width = 832

    video = engine.run(
        height=height,
        width=width,
        prompt=prompt,
        # negative_prompt=negative_prompt,
        use_cfg_guidance=False,
        duration="5s",
        num_videos=1,
        guidance_scale=5.0,
        seed=420,
    )

    export_to_video(video[0], "t2v_420.mp4", fps=16, quality=8)
