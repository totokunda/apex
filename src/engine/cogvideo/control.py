import torch
from typing import Dict, Any, Callable, List, Union, Optional
from PIL import Image
import numpy as np

from .base import CogVideoBaseEngine


class CogVideoControlEngine(CogVideoBaseEngine):
    """CogVideo Control Engine Implementation"""

    def run(
        self,
        prompt: Union[List[str], str],
        control_video: Union[List[Image.Image], torch.Tensor],
        negative_prompt: Union[List[str], str] = None,
        height: int = 480,
        width: int = 720,
        num_inference_steps: int = 50,
        num_videos: int = 1,
        seed: int = None,
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
        **kwargs,
    ):
        """Control video generation following CogVideoXFunControlPipeline"""

        # 1. Process control video
        if isinstance(control_video[0], Image.Image):
            control_video = [control_video]

        control_video = self.video_processor.preprocess_video(
            control_video, height=height, width=width
        )
        control_video = control_video.to(device=self.device)

        num_frames = len(control_video[0])

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

        # 6. Prepare latents
        latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1

        # For CogVideoX 1.5, check that latent frames are divisible by patch_size_t
        patch_size_t = getattr(self.transformer.config, "patch_size_t", None)
        if patch_size_t is not None and latent_frames % patch_size_t != 0:
            raise ValueError(
                f"The number of latent frames must be divisible by `{patch_size_t=}` but the given video "
                f"contains {latent_frames=}, which is not divisible."
            )

        # Prepare control video latents using vae_encode
        control_video = control_video.to(device=self.device, dtype=prompt_embeds.dtype)
        control_video_latents = self.vae_encode(
            control_video, sample_mode="mode", dtype=prompt_embeds.dtype
        )
        control_video_latents = control_video_latents.permute(0, 2, 1, 3, 4)

        # Prepare noise latents
        latent_channels = self.transformer.config.in_channels // 2
        latents = self._get_latents(
            height,
            width,
            num_frames,
            num_videos=num_videos,
            num_channels_latents=latent_channels,
            seed=seed,
            generator=generator,
            dtype=prompt_embeds.dtype,
        )

        # Scale initial noise by scheduler's init_noise_sigma
        latents = latents * self.scheduler.init_noise_sigma

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
            control_video_latents=control_video_latents,
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
