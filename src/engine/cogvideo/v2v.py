import torch
from typing import Dict, Any, Callable, List, Union, Optional
from PIL import Image
import numpy as np

from .base import CogVideoBaseEngine


class CogVideoV2VEngine(CogVideoBaseEngine):
    """CogVideo Video-to-Video Engine Implementation"""

    def run(
        self,
        video: Union[List[Image.Image], torch.Tensor],
        prompt: Union[List[str], str],
        negative_prompt: Union[List[str], str] = None,
        height: int = 480,
        width: int = 720,
        num_inference_steps: int = 50,
        num_videos: int = 1,
        seed: int = None,
        strength: float = 0.8,
        guidance_scale: float = 6.0,
        use_dynamic_cfg: bool = False,
        return_latents: bool = False,
        text_encoder_kwargs: Dict[str, Any] = {},
        attention_kwargs: Dict[str, Any] = {},
        render_on_step_callback: Callable = None,
        offload: bool = True,
        render_on_step: bool = False,
        generator: torch.Generator = None,
        timesteps: List[int] = None,
        max_sequence_length: int = 226,
        latents: torch.Tensor = None,
        **kwargs,
    ):
        """Video-to-video generation following CogVideoXVideoToVideoPipeline"""

        # 1. Process input video
        if latents is None:
            video = self.video_processor.preprocess_video(
                video, height=height, width=width
            )
            video = video.to(device=self.device)

        num_frames = len(video) if latents is None else latents.size(1)

        # 2. Encode prompts
        prompt_embeds, negative_prompt_embeds = self._encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_videos_per_prompt=num_videos,
            max_sequence_length=max_sequence_length,
            **text_encoder_kwargs,
        )

        if offload:
            self._offload(self.text_encoder)

        # 3. Load transformer
        if not self.transformer:
            self.load_component_by_type("transformer")

        self.to_device(self.transformer)
        transformer_dtype = self.component_dtypes["transformer"]

        prompt_embeds = prompt_embeds.to(self.device, dtype=transformer_dtype)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(
                self.device, dtype=transformer_dtype
            )

        # 4. Load scheduler
        if not self.scheduler:
            self.load_component_by_type("scheduler")
        self.to_device(self.scheduler)

        # 5. Prepare timesteps
        timesteps, num_inference_steps = self._get_timesteps(
            num_inference_steps=num_inference_steps,
            timesteps=timesteps,
        )

        # Get timesteps for video-to-video (strength-based)
        timesteps, num_inference_steps = self._get_v2v_timesteps(
            num_inference_steps, timesteps, strength
        )
        latent_timestep = timesteps[:1].repeat(num_videos)

        # 6. Prepare latents
        latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1

        # For CogVideoX 1.5, check that latent frames are divisible by patch_size_t
        patch_size_t = getattr(self.transformer.config, "patch_size_t", None)
        if patch_size_t is not None and latent_frames % patch_size_t != 0:
            raise ValueError(
                f"The number of latent frames must be divisible by `{patch_size_t=}` but the given video "
                f"contains {latent_frames=}, which is not divisible."
            )

        # Prepare latents from input video
        if latents is None:
            video = video.to(dtype=prompt_embeds.dtype, device=self.device)
            latents = self._prepare_v2v_latents(
                video=video,
                batch_size=num_videos,
                num_channels_latents=getattr(
                    self.transformer.config, "in_channels", 16
                ),
                height=height,
                width=width,
                dtype=prompt_embeds.dtype,
                device=self.device,
                generator=generator,
                timestep=latent_timestep,
            )

        # 7. Prepare rotary embeddings
        image_rotary_emb = self._prepare_rotary_positional_embeddings(
            height, width, latents.size(1), self.device
        )

        # 8. Prepare guidance
        do_classifier_free_guidance = (
            guidance_scale > 1.0 and negative_prompt_embeds is not None
        )
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)

        # 9. Denoising loop
        latents = self.denoise(
            latents=latents,
            timesteps=timesteps,
            scheduler=self.scheduler,
            guidance_scale=guidance_scale,
            use_dynamic_cfg=use_dynamic_cfg,
            do_classifier_free_guidance=do_classifier_free_guidance,
            noise_pred_kwargs=dict(
                encoder_hidden_states=prompt_embeds,
                image_rotary_emb=image_rotary_emb,
                attention_kwargs=attention_kwargs,
            ),
            render_on_step=render_on_step,
            render_on_step_callback=render_on_step_callback,
            num_inference_steps=num_inference_steps,
            **kwargs,
        )

        if offload:
            self._offload(self.transformer)

        if return_latents:
            return latents
        else:
            video = self.vae_decode(latents, offload=offload)
            postprocessed_video = self._postprocess(video)
            return postprocessed_video
