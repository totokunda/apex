import torch
from typing import Dict, Any, Callable, List, Union, Optional
from PIL import Image
import numpy as np
from .shared import WanShared
from src.utils.progress import safe_emit_progress, make_mapped_progress


class WanT2IEngine(WanShared):
    """WAN Text-to-Image Engine Implementation"""

    def run(
        self,
        prompt: List[str] | str,
        negative_prompt: List[str] | str = None,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 30,
        num_images: int = 1,
        seed: int | None = None,
        guidance_scale: float = 5.0,
        high_noise_guidance_scale: float | None = None,
        low_noise_guidance_scale: float | None = None,
        use_cfg_guidance: bool = True,
        return_latents: bool = False,
        text_encoder_kwargs: Dict[str, Any] = {},
        attention_kwargs: Dict[str, Any] = {},
        render_on_step_callback: Callable = None,
        render_on_step_interval: int = 3,
        offload: bool = True,
        render_on_step: bool = False,
        generator: torch.Generator | None = None,
        timesteps: List[int] | None = None,
        timesteps_as_indices: bool = True,
        boundary_ratio: float | None = 0.875,
        expand_timesteps: bool = False,
        ip_image: Image.Image | str | np.ndarray | torch.Tensor = None,
        enhance_kwargs: Dict[str, Any] = {},
        progress_callback: Callable = None,
        denoise_progress_callback: Callable = None,
        **kwargs,
    ):

        safe_emit_progress(progress_callback, 0.0, "Starting t2i pipeline")
        
        if high_noise_guidance_scale is not None and low_noise_guidance_scale is not None:
            guidance_scale = [high_noise_guidance_scale, low_noise_guidance_scale]

        if not self.text_encoder:
            self.load_component_by_type("text_encoder")

        self.to_device(self.text_encoder)

        prompt_embeds = self.text_encoder.encode(
            prompt,
            device=self.device,
            num_videos_per_prompt=num_images,
            **text_encoder_kwargs,
        )
        
        safe_emit_progress(progress_callback, 0.10, "Encoded prompt")
        
        batch_size = prompt_embeds.shape[0]

        if negative_prompt is not None and use_cfg_guidance:
            negative_prompt_embeds = self.text_encoder.encode(
                negative_prompt,
                device=self.device,
                num_videos_per_prompt=num_images,
                **text_encoder_kwargs,
            )
        else:
            negative_prompt_embeds = None
        safe_emit_progress(
            progress_callback,
            0.14,
            "Prepared negative prompt" if negative_prompt_embeds is not None else "Skipped negative prompt",
        )

        if offload:
            self._offload(self.text_encoder)
        safe_emit_progress(progress_callback, 0.16, "Text encoder offloaded")

        transformer_dtype = self.component_dtypes["transformer"]
        prompt_embeds = prompt_embeds.to(self.device, dtype=transformer_dtype)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(
                self.device, dtype=transformer_dtype
            )
        safe_emit_progress(progress_callback, 0.20, "Prepared embeddings for transformer")

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
        safe_emit_progress(progress_callback, 0.28, "Scheduler and timesteps prepared")

        vae_config = self.load_config_by_type("vae")
        vae_scale_factor_spatial = getattr(
            vae_config, "scale_factor_spatial", self.vae_scale_factor_spatial
        )
        vae_scale_factor_temporal = getattr(
            vae_config, "scale_factor_temporal", self.vae_scale_factor_temporal
        )

        latents = self._get_latents(
            height,
            width,
            1,
            num_channels_latents=getattr(vae_config, "z_dim", 16),
            vae_scale_factor_spatial=vae_scale_factor_spatial,
            vae_scale_factor_temporal=vae_scale_factor_temporal,
            fps=16,
            batch_size=batch_size,
            seed=seed,
            dtype=torch.float32,
            generator=generator,
        )
        safe_emit_progress(progress_callback, 0.36, "Initialized latent noise")

        if boundary_ratio is not None:
            boundary_timestep = boundary_ratio * getattr(
                self.scheduler.config, "num_train_timesteps", 1000
            )
        else:
            boundary_timestep = None

        # Set preview context for step-wise rendering
        
        self._preview_height = height
        self._preview_width = width
        self._preview_offload = offload
        # Reserve denoising progress range
        denoise_progress_callback = make_mapped_progress(progress_callback, 0.40, 0.92)
        safe_emit_progress(progress_callback, 0.40, "Starting denoising")

        latents = self.denoise(
            expand_timesteps=expand_timesteps,
            boundary_timestep=boundary_timestep,
            timesteps=timesteps,
            latents=latents,
            latent_condition=None,
            transformer_kwargs=dict(
                encoder_hidden_states=prompt_embeds,
                attention_kwargs=attention_kwargs,
                enhance_kwargs=enhance_kwargs,
            ),
            unconditional_transformer_kwargs=(
                dict(
                    encoder_hidden_states=negative_prompt_embeds,
                    attention_kwargs=attention_kwargs,
                    enhance_kwargs=enhance_kwargs,
                )
                if negative_prompt_embeds is not None
                else None
            ),
            transformer_dtype=transformer_dtype,
            use_cfg_guidance=use_cfg_guidance,
            render_on_step=render_on_step,
            render_on_step_callback=render_on_step_callback,
            render_on_step_interval=render_on_step_interval,
            scheduler=scheduler,
            guidance_scale=guidance_scale,
            ip_image=ip_image,
            denoise_progress_callback=denoise_progress_callback,
        )
        safe_emit_progress(progress_callback, 0.94, "Denoising complete")

        if offload:
            self._offload(self.transformer)
        safe_emit_progress(progress_callback, 0.96, "Transformer offloaded")

        if return_latents:
            safe_emit_progress(progress_callback, 1.0, "Returning latents")
            return latents
        else:
            tensor_image = self.vae_decode(latents, offload=offload)
            image = self._tensor_to_frame(tensor_image)
            safe_emit_progress(progress_callback, 1.0, "Completed t2i pipeline")
            return image
    
    
