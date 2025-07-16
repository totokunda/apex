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
from src.preprocess.base.camera import Camera
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

    def t2v_run(
        self,
        prompt: List[str] | str,
        negative_prompt: List[str] | str = None,
        height: int = 480,
        width: int = 832,
        duration: int | str = 16,
        num_inference_steps: int = 30,
        num_videos: int = 1,
        seed: int | None = None,
        fps: int = 16,
        guidance_scale: float = 1.0,
        use_cfg_guidance: bool = True,
        return_latents: bool = False,
        text_encoder_kwargs: Dict[str, Any] = {},
        attention_kwargs: Dict[str, Any] = {},
        render_on_step_callback: Callable = None,
        offload: bool = True,
        render_on_step: bool = False,
        generator: torch.Generator | None = None,
        timesteps: List[int] | None = None,
        timesteps_as_indices: bool = True,
        **kwargs,
    ):

        if not self.text_encoder:
            self.load_component_by_type("text_encoder")

        self.to_device(self.text_encoder)

        use_cfg_guidance = (
            use_cfg_guidance and negative_prompt is not None and guidance_scale > 1.0
        )

        prompt_embeds = self.text_encoder.encode(
            prompt,
            device=self.device,
            num_videos_per_prompt=num_videos,
            **text_encoder_kwargs,
        )

        if negative_prompt is not None and use_cfg_guidance:
            negative_prompt_embeds = self.text_encoder.encode(
                negative_prompt,
                device=self.device,
                num_videos_per_prompt=num_videos,
                **text_encoder_kwargs,
            )
        else:
            negative_prompt_embeds = None

        if offload:
            self._offload(self.text_encoder)

        if "step1_text_encoder" not in self.preprocessors:
            self.load_preprocessor_by_type("step1_text_encoder")

        step1_text_encoder = self.preprocessors["step1_text_encoder"]
        llm_prompt_embeds, llm_mask = step1_text_encoder(
            prompt, with_mask=True, max_length=320
        )

        if use_cfg_guidance:
            llm_negative_prompt_embeds, llm_negative_mask = step1_text_encoder(
                negative_prompt, with_mask=True, max_length=320
            )

        if not self.transformer:
            self.load_component_by_type("transformer")

        self.to_device(self.transformer)
        transformer_dtype = self.component_dtypes["transformer"]
        prompt_embeds = prompt_embeds.to(self.device, dtype=transformer_dtype)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(
                self.device, dtype=transformer_dtype
            )

        if not self.scheduler:
            self.load_component_by_type("scheduler")

        self.to_device(self.scheduler)
        scheduler = self.scheduler

        scheduler.set_timesteps(
            num_inference_steps if timesteps is None else 1000, device=self.device
        )

        timesteps, num_inference_steps = self._get_timesteps(
            scheduler=scheduler,
            timesteps=timesteps,
            timesteps_as_indices=timesteps_as_indices,
            num_inference_steps=num_inference_steps,
        )

        num_frames = self._parse_num_frames(duration, fps)
        ## ensure its divisible by 17
        latent_num_frames = max(num_frames // 17 * 3, 17)

        latents = self._get_latents(
            height,
            width,
            latent_num_frames,
            fps=fps,
            num_videos=num_videos,
            seed=seed,
            dtype=torch.float32,
            generator=generator,
            parse_frames=False,
        )

        latents = self.denoise(
            timesteps=timesteps,
            latents=latents,
            latent_condition=None,
            transformer_kwargs=dict(
                encoder_hidden_states=prompt_embeds,
                encoder_hidden_states_2=llm_prompt_embeds,
                encoder_attention_mask=llm_mask,
                attention_kwargs=attention_kwargs,
            ),
            unconditional_transformer_kwargs=(
                dict(
                    encoder_hidden_states=negative_prompt_embeds,
                    encoder_hidden_states_2=llm_negative_prompt_embeds,
                    encoder_attention_mask=llm_negative_mask,
                    attention_kwargs=attention_kwargs,
                )
                if negative_prompt_embeds is not None
                else None
            ),
            transformer_dtype=transformer_dtype,
            use_cfg_guidance=use_cfg_guidance,
            render_on_step=render_on_step,
            render_on_step_callback=render_on_step_callback,
            scheduler=scheduler,
            guidance_scale=guidance_scale,
        )

        if offload:
            self._offload(self.transformer)

        if return_latents:
            return latents
        else:
            video = self.vae_decode(latents, offload=offload)
            postprocessed_video = self._postprocess(video)
            return postprocessed_video

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
