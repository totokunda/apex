from src.engine.base_engine import BaseEngine
import torch
from typing import Dict, Any, Callable, List, Union, Optional
from enum import Enum
from src.ui.nodes import UINode
from diffusers.video_processor import VideoProcessor
import math
from PIL import Image
import numpy as np
from src.mixins import OffloadMixin
import torchvision.transforms.functional as TF
from src.engine.denoise.wan_denoise import WanDenoise, DenoiseType
from src.preprocess.camera.camera import Camera
import torch.nn.functional as F


class ModelType(Enum):
    VACE = "vace"  # vace
    T2V = "t2v"  # text to video
    I2V = "i2v"  # image to video
    FFLF = "fflf"  # first frame last frame
    CAUSAL = "causal"  # causal
    FUN = "fun"  # fun
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
        super().__init__(yaml_path, **kwargs)

        self.model_type = model_type
        self.denoise_type = denoise_type

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
        elif self.model_type == ModelType.VACE:
            return self.vace_run(**final_kwargs)
        elif self.model_type == ModelType.I2V:
            return self.i2v_run(**final_kwargs)
        elif self.model_type == ModelType.FFLF:
            return self.fflf_run(**final_kwargs)
        elif self.model_type == ModelType.CAUSAL:
            return self.causal_run(**final_kwargs)
        elif self.model_type == ModelType.FUN:
            return self.fun_run(**final_kwargs)
        elif self.model_type == ModelType.APEX_FRAMEPACK:
            return self.apex_framepack_run(**final_kwargs)
        elif self.model_type == ModelType.MULTITALK:
            return self.multitalk_run(**final_kwargs)
        else:
            raise ValueError(f"Invalid model type: {self.model_type}")

    def _prepare_fun_control_latents(
        self, control, dtype=torch.float32, generator: torch.Generator | None = None
    ):
        # resize the control to latents shape as we concatenate the control to the latents
        # we do that before converting to dtype to avoid breaking in case we're using cpu_offload
        # and half precision

        control = control.to(device=self.device, dtype=dtype)
        bs = 1
        new_control = []
        for i in range(0, control.shape[0], bs):
            control_bs = control[i : i + bs]
            control_bs = self.vae_encode(
                control_bs, sample_generator=generator, normalize_latents_dtype=dtype
            )
            new_control.append(control_bs)
        control = torch.cat(new_control, dim=0)

        return control

    def _apply_color_correction(
        self, videos: torch.Tensor, reference_image: torch.Tensor, strength: float
    ) -> torch.Tensor:
        """Apply color correction to match reference image."""
        # Simple color matching - can be enhanced with more sophisticated methods
        # This is a placeholder implementation
        return videos

    def __str__(self):
        return f"WanEngine(config={self.config}, device={self.device}, model_type={self.model_type})"

    def __repr__(self):
        return self.__str__()


if __name__ == "__main__":
    from diffusers.utils import export_to_video
    from PIL import Image

    engine = WanEngine(
        yaml_path="manifest/wan_t2v_sf_1.3b.yml",
        model_type=ModelType.CAUSAL,
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

    export_to_video(video[0], "t2v_1.3b_sf_420.mp4", fps=16, quality=8)
