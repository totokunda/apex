import torch
from enum import Enum
from typing import Optional


class DenoiseType(Enum):
    T2V = "t2v"
    I2V = "i2v"
    CONDITION = "condition"


def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    r"""
    Rescales `noise_cfg` tensor based on `guidance_rescale` to improve image quality and fix overexposure. Based on
    Section 3.4 from [Common Diffusion Noise Schedules and Sample Steps are
    Flawed](https://huggingface.co/papers/2305.08891).

    Args:
        noise_cfg (`torch.Tensor`):
            The predicted noise tensor for the guided diffusion process.
        noise_pred_text (`torch.Tensor`):
            The predicted noise tensor for the text-guided diffusion process.
        guidance_rescale (`float`, *optional*, defaults to 0.0):
            A rescale factor applied to the noise predictions.

    Returns:
        noise_cfg (`torch.Tensor`): The rescaled noise prediction tensor.
    """
    std_text = noise_pred_text.std(
        dim=list(range(1, noise_pred_text.ndim)), keepdim=True
    )
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = (
        guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    )
    return noise_cfg


class LTXDenoise:
    def __init__(self, denoise_type: DenoiseType = DenoiseType.T2V, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.denoise_type = denoise_type

    def denoise(self, *args, **kwargs) -> torch.Tensor:
        if self.denoise_type == DenoiseType.T2V:
            return self.t2v_denoise(*args, **kwargs)
        elif self.denoise_type == DenoiseType.I2V:
            return self.i2v_denoise(*args, **kwargs)
        elif self.denoise_type == DenoiseType.CONDITION:
            return self.condition_denoise(*args, **kwargs)
        else:
            raise ValueError(f"Denoise type {self.denoise_type} not supported")

    def t2v_denoise(self, *args, **kwargs) -> torch.Tensor:
        timesteps = kwargs.get("timesteps", None)
        latents = kwargs.get("latents", None)
        transformer_dtype = kwargs.get("transformer_dtype", None)
        use_cfg_guidance = kwargs.get("use_cfg_guidance", True)
        render_on_step = kwargs.get("render_on_step", False)
        render_on_step_callback = kwargs.get("render_on_step_callback", None)
        scheduler = kwargs.get("scheduler", None)
        guidance_scale = kwargs.get("guidance_scale", 5.0)
        num_inference_steps = kwargs.get("num_inference_steps", 50)
        prompt_embeds = kwargs.get("prompt_embeds", None)
        prompt_attention_mask = kwargs.get("prompt_attention_mask", None)
        latent_num_frames = kwargs.get("latent_num_frames", None)
        latent_height = kwargs.get("latent_height", None)
        latent_width = kwargs.get("latent_width", None)
        rope_interpolation_scale = kwargs.get("rope_interpolation_scale", None)
        attention_kwargs = kwargs.get("attention_kwargs", None)
        guidance_rescale = kwargs.get("guidance_rescale", 0.0)
        num_warmup_steps = kwargs.get("num_warmup_steps", 0)

        with self._progress_bar(
            total=num_inference_steps, desc="Denoising T2V"
        ) as pbar:
            for i, t in enumerate(timesteps):

                latent_model_input = (
                    torch.cat([latents] * 2) if use_cfg_guidance else latents
                )

                latent_model_input = latent_model_input.to(transformer_dtype)

                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latent_model_input.shape[0])

                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    encoder_hidden_states=prompt_embeds,
                    timestep=timestep,
                    encoder_attention_mask=prompt_attention_mask,
                    num_frames=latent_num_frames,
                    height=latent_height,
                    width=latent_width,
                    rope_interpolation_scale=rope_interpolation_scale,
                    attention_kwargs=attention_kwargs,
                    return_dict=False,
                )[0]

                noise_pred = noise_pred.float()

                if use_cfg_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )

                    if guidance_rescale > 0:
                        noise_pred = rescale_noise_cfg(
                            noise_pred,
                            noise_pred_text,
                            guidance_rescale=guidance_rescale,
                        )

                # compute the previous noisy sample x_t -> x_t-1
                latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if render_on_step and render_on_step_callback:
                    self._render_step(latents, render_on_step_callback)

                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % scheduler.order == 0
                ):
                    pbar.update(1)

            self.logger.info("Denoising completed.")

        return latents

    def i2v_denoise(self, *args, **kwargs) -> torch.Tensor:

        timesteps = kwargs.get("timesteps", None)
        latents = kwargs.get("latents", None)
        transformer_dtype = kwargs.get("transformer_dtype", None)
        use_cfg_guidance = kwargs.get("use_cfg_guidance", True)
        render_on_step = kwargs.get("render_on_step", False)
        render_on_step_callback = kwargs.get("render_on_step_callback", None)
        scheduler = kwargs.get("scheduler", None)
        guidance_scale = kwargs.get("guidance_scale", 5.0)
        num_inference_steps = kwargs.get("num_inference_steps", 50)
        prompt_embeds = kwargs.get("prompt_embeds", None)
        prompt_attention_mask = kwargs.get("prompt_attention_mask", None)
        latent_num_frames = kwargs.get("latent_num_frames", None)
        latent_height = kwargs.get("latent_height", None)
        latent_width = kwargs.get("latent_width", None)
        rope_interpolation_scale = kwargs.get("rope_interpolation_scale", None)
        attention_kwargs = kwargs.get("attention_kwargs", None)
        guidance_rescale = kwargs.get("guidance_rescale", 0.0)
        num_warmup_steps = kwargs.get("num_warmup_steps", 0)
        conditioning_mask = kwargs.get("conditioning_mask", None)

        with self._progress_bar(
            total=num_inference_steps, desc="Denoising I2V"
        ) as pbar:
            for i, t in enumerate(timesteps):

                latent_model_input = (
                    torch.cat([latents] * 2) if use_cfg_guidance else latents
                )
                latent_model_input = latent_model_input.to(transformer_dtype)

                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latent_model_input.shape[0])
                timestep = timestep.unsqueeze(-1) * (1 - conditioning_mask)

                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    encoder_hidden_states=prompt_embeds,
                    timestep=timestep,
                    encoder_attention_mask=prompt_attention_mask,
                    num_frames=latent_num_frames,
                    height=latent_height,
                    width=latent_width,
                    rope_interpolation_scale=rope_interpolation_scale,
                    attention_kwargs=attention_kwargs,
                    return_dict=False,
                )[0]

                noise_pred = noise_pred.float()

                if use_cfg_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )

                    if guidance_rescale > 0:
                        noise_pred = rescale_noise_cfg(
                            noise_pred,
                            noise_pred_text,
                            guidance_rescale=guidance_rescale,
                        )

                noise_pred = self._unpack_latents(
                    noise_pred,
                    latent_num_frames,
                    latent_height,
                    latent_width,
                    self.transformer_spatial_patch_size,
                    self.transformer_temporal_patch_size,
                )
                latents = self._unpack_latents(
                    latents,
                    latent_num_frames,
                    latent_height,
                    latent_width,
                    self.transformer_spatial_patch_size,
                    self.transformer_temporal_patch_size,
                )

                noise_pred = noise_pred[:, :, 1:]
                noise_latents = latents[:, :, 1:]

                pred_latents = self.scheduler.step(
                    noise_pred, t, noise_latents, return_dict=False
                )[0]

                latents = torch.cat([latents[:, :, :1], pred_latents], dim=2)
                latents = self._pack_latents(
                    latents,
                    self.transformer_spatial_patch_size,
                    self.transformer_temporal_patch_size,
                )

                if render_on_step and render_on_step_callback:
                    self._render_step(latents, render_on_step_callback)

                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % scheduler.order == 0
                ):
                    pbar.update(1)

            self.logger.info("Denoising completed.")

        return latents

    def condition_denoise(self, *args, **kwargs) -> torch.Tensor:
        timesteps = kwargs.get("timesteps", None)
        latents = kwargs.get("latents", None)
        transformer_dtype = kwargs.get("transformer_dtype", None)
        use_cfg_guidance = kwargs.get("use_cfg_guidance", True)
        render_on_step = kwargs.get("render_on_step", False)
        render_on_step_callback = kwargs.get("render_on_step_callback", None)
        scheduler = kwargs.get("scheduler", None)
        guidance_scale = kwargs.get("guidance_scale", 5.0)
        num_inference_steps = kwargs.get("num_inference_steps", 50)
        prompt_embeds = kwargs.get("prompt_embeds", None)
        prompt_attention_mask = kwargs.get("prompt_attention_mask", None)
        attention_kwargs = kwargs.get("attention_kwargs", None)
        guidance_rescale = kwargs.get("guidance_rescale", 0.0)
        num_warmup_steps = kwargs.get("num_warmup_steps", 0)
        conditioning_mask = kwargs.get("conditioning_mask", None)
        image_cond_noise_scale = kwargs.get("image_cond_noise_scale", 0.0)
        generator = kwargs.get("generator", None)
        use_cfg_guidance = kwargs.get("use_cfg_guidance", True)
        guidance_scale = kwargs.get("guidance_scale", 5.0)
        guidance_rescale = kwargs.get("guidance_rescale", 0.0)
        num_inference_steps = kwargs.get("num_inference_steps", 50)
        prompt_embeds = kwargs.get("prompt_embeds", None)
        init_latents = kwargs.get("init_latents", None)
        video_coords = kwargs.get("video_coords", None)
        is_conditioning_image_or_video = kwargs.get(
            "is_conditioning_image_or_video", False
        )

        with self._progress_bar(
            total=num_inference_steps, desc="Denoising CONDITION"
        ) as progress_bar:
            for i, t in enumerate(timesteps):
                if image_cond_noise_scale > 0 and init_latents is not None:
                    # Add timestep-dependent noise to the hard-conditioning latents
                    # This helps with motion continuity, especially when conditioned on a single frame
                    latents = self.add_noise_to_image_conditioning_latents(
                        t / 1000.0,
                        init_latents,
                        latents,
                        image_cond_noise_scale,
                        conditioning_mask,
                        generator,
                    )

                latent_model_input = (
                    torch.cat([latents] * 2) if use_cfg_guidance else latents
                )
                if is_conditioning_image_or_video:
                    conditioning_mask_model_input = (
                        torch.cat([conditioning_mask, conditioning_mask])
                        if self.do_classifier_free_guidance
                        else conditioning_mask
                    )
                latent_model_input = latent_model_input.to(transformer_dtype)

                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latent_model_input.shape[0]).unsqueeze(-1).float()
                if is_conditioning_image_or_video:
                    timestep = torch.min(
                        timestep, (1 - conditioning_mask_model_input) * 1000.0
                    )

                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    encoder_hidden_states=prompt_embeds,
                    timestep=timestep,
                    encoder_attention_mask=prompt_attention_mask,
                    video_coords=video_coords,
                    attention_kwargs=attention_kwargs,
                    return_dict=False,
                )[0]

                if use_cfg_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )
                    timestep, _ = timestep.chunk(2)

                    if guidance_rescale > 0:
                        # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                        noise_pred = rescale_noise_cfg(
                            noise_pred,
                            noise_pred_text,
                            guidance_rescale=guidance_rescale,
                        )

                denoised_latents = scheduler.step(
                    -noise_pred,
                    t,
                    latents,
                    per_token_timesteps=timestep,
                    return_dict=False,
                )[0]
                if is_conditioning_image_or_video:
                    tokens_to_denoise_mask = (
                        t / 1000 - 1e-6 < (1.0 - conditioning_mask)
                    ).unsqueeze(-1)
                    latents = torch.where(
                        tokens_to_denoise_mask, denoised_latents, latents
                    )
                else:
                    latents = denoised_latents

                if render_on_step and render_on_step_callback:
                    self._render_step(latents, render_on_step_callback)

                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % scheduler.order == 0
                ):
                    progress_bar.update()

        return latents
