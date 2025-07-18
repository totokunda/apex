from .base import LTXBaseEngine
from typing import Dict, Any, Callable, List, Union, Optional
import torch
import numpy as np
from PIL import Image

class LTXI2VEngine(LTXBaseEngine):
    """LTX Image-to-Video Engine Implementation"""
    
    def run(
        self,
        image: Union[Image.Image, str, np.ndarray, torch.Tensor],
        prompt: List[str] | str,
        negative_prompt: List[str] | str = None,
        height: int = 480,
        width: int = 832,
        duration: int | str = 16,
        num_inference_steps: int = 30,
        num_videos: int = 1,
        seed: int | None = None,
        fps: int = 25,
        use_cfg_guidance: bool = True,
        text_encoder_kwargs: Dict[str, Any] = {},
        guidance_scale: float = 5.0,
        guidance_rescale: float = 0.0,
        offload: bool = True,
        render_on_step: bool = False,
        generator: torch.Generator | None = None,
        timesteps: List[int] | None = None,
        render_on_step_callback: Callable = None,
        attention_kwargs: Dict[str, Any] = {},
        return_latents: bool = False,
        decode_timestep: Union[float, List[float]] = 0.0,
        decode_noise_scale: Optional[Union[float, List[float]]] = None,
        **kwargs,
    ):
        postprocessor_kwargs = kwargs.get("postprocessor_kwargs", None)
        if not self.text_encoder:
            self.load_component_by_type("text_encoder")

        self.to_device(self.text_encoder)

        if seed is not None and generator is None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        elif seed is not None and generator is not None:
            self.logger.warning(
                "Both seed and generator are provided. Ignoring seed and using generator."
            )
            seed = None

        prompt_embeds, prompt_attention_mask = self.text_encoder.encode(
            prompt,
            device=self.device,
            num_videos_per_prompt=num_videos,
            return_attention_mask=True,
            **text_encoder_kwargs,
        )

        if negative_prompt is not None and use_cfg_guidance:
            negative_prompt_embeds, negative_prompt_attention_mask = (
                self.text_encoder.encode(
                    negative_prompt,
                    device=self.device,
                    num_videos_per_prompt=num_videos,
                    return_attention_mask=True,
                    **text_encoder_kwargs,
                )
            )
        else:
            negative_prompt_embeds, negative_prompt_attention_mask = torch.zeros_like(
                prompt_embeds
            ), torch.zeros_like(prompt_attention_mask)

        if use_cfg_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            prompt_attention_mask = torch.cat(
                [negative_prompt_attention_mask, prompt_attention_mask], dim=0
            )

        if offload:
            self._offload(self.text_encoder)

        if not self.transformer:
            self.load_component_by_type("transformer")

        self.to_device(self.transformer)
        transformer_dtype = self.component_dtypes["transformer"]

        if image is not None:
            loaded_image = self._load_image(image)
            # make height divisible by vae_scale_factor_spatial
            height = (
                height // self.vae_scale_factor_spatial * self.vae_scale_factor_spatial
            )
            width = (
                width // self.vae_scale_factor_spatial * self.vae_scale_factor_spatial
            )

            prepocessed_image = self.video_processor.preprocess(
                loaded_image, height=height, width=width
            )
            prepocessed_image = prepocessed_image.to(
                device=self.device, dtype=transformer_dtype
            ).unsqueeze(2)

        num_frames = self._parse_num_frames(duration, fps)
        latent_num_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        latent_height = height // self.vae_scale_factor_spatial
        latent_width = width // self.vae_scale_factor_spatial
        video_sequence_length = latent_num_frames * latent_height * latent_width
        batch_size = num_videos

        init_latents = self.vae_encode(
            prepocessed_image,
            sample_generator=generator,
            sample_mode="sample",
            dtype=torch.float32,
            offload=offload,
        ).repeat(1, 1, latent_num_frames, 1, 1)

        conditioning_mask = torch.zeros(
            batch_size,
            1,
            latent_num_frames,
            latent_height,
            latent_width,
            device=self.device,
            dtype=torch.float32,
        )

        conditioning_mask[:, :, 0] = 1.0

        noise_latents = self._get_latents(
            height=height,
            width=width,
            duration=latent_num_frames,
            fps=fps,
            num_videos=num_videos,
            dtype=torch.float32,
            seed=seed,
            generator=generator,
            parse_frames=False,
        )

        latents = init_latents * conditioning_mask + noise_latents * (
            1 - conditioning_mask
        )

        conditioning_mask = self._pack_latents(
            conditioning_mask,
            self.transformer_spatial_patch_size,
            self.transformer_temporal_patch_size,
        ).squeeze(-1)

        latents = self._pack_latents(
            latents,
            self.transformer_spatial_patch_size,
            self.transformer_temporal_patch_size,
        )

        prompt_embeds = prompt_embeds.to(device=self.device, dtype=transformer_dtype)
        prompt_attention_mask = prompt_attention_mask.to(device=self.device)

        if not self.scheduler:
            self.load_component_by_type("scheduler")

        self.to_device(self.scheduler)

        scheduler = self.scheduler

        if use_cfg_guidance:
            conditioning_mask = torch.cat([conditioning_mask, conditioning_mask])

        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        mu = self._calculate_shift(
            video_sequence_length,
            scheduler.config.get("base_image_seq_len", 256),
            scheduler.config.get("max_image_seq_len", 4096),
            scheduler.config.get("base_shift", 0.5),
            scheduler.config.get("max_shift", 1.15),
        )

        timesteps, num_inference_steps = self._get_timesteps(
            scheduler,
            num_inference_steps,
            self.device,
            timesteps,
            mu=mu,
            sigmas=sigmas if not timesteps else None,
        )

        num_warmup_steps = max(
            len(timesteps) - num_inference_steps * scheduler.order, 0
        )

        # 6. Prepare micro-conditions
        rope_interpolation_scale = (
            self.vae_scale_factor_temporal / fps,
            self.vae_scale_factor_spatial,
            self.vae_scale_factor_spatial,
        )

        latents = self.denoise(
            latents=latents,
            timesteps=timesteps,
            num_warmup_steps=num_warmup_steps,
            num_inference_steps=num_inference_steps,
            num_videos=num_videos,
            seed=seed,
            fps=fps,
            prompt_embeds=prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            latent_num_frames=latent_num_frames,
            latent_height=latent_height,
            latent_width=latent_width,
            rope_interpolation_scale=rope_interpolation_scale,
            attention_kwargs=attention_kwargs,
            guidance_rescale=guidance_rescale,
            use_cfg_guidance=use_cfg_guidance,
            guidance_scale=guidance_scale,
            render_on_step=render_on_step,
            transformer_dtype=transformer_dtype,
            render_on_step_callback=render_on_step_callback,
            conditioning_mask=conditioning_mask,
            scheduler=scheduler,
        )

        return self.prepare_output(
            latents=latents,
            latent_num_frames=latent_num_frames,
            latent_height=latent_height,
            latent_width=latent_width,
            offload=offload,
            return_latents=return_latents,
            generator=generator,
            decode_timestep=decode_timestep,
            decode_noise_scale=decode_noise_scale,
            postprocessor_kwargs=postprocessor_kwargs,
        )