import torch
from src.utils.type import EnumType
from typing import Optional
import torch
from diffusers.utils.torch_utils import randn_tensor


class DenoiseType(EnumType):
    BASE = "base"


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
    def __init__(self, denoise_type: DenoiseType = DenoiseType.BASE, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.denoise_type = denoise_type

    def denoise(self, *args, **kwargs) -> torch.Tensor:
        if self.denoise_type == DenoiseType.BASE:
            return self.base_denoise(*args, **kwargs)
        else:
            raise ValueError(f"Denoise type {self.denoise_type} not supported")

    def base_denoise(self, *args, **kwargs) -> torch.Tensor:
        conditioning_mask = kwargs.get("conditioning_mask", None)
        latents = kwargs.get("latents", None)
        timesteps = kwargs.get("timesteps", None)
        num_inference_steps = kwargs.get("num_inference_steps", 50)
        prompt_embeds_batch = kwargs.get("prompt_embeds_batch", None)
        prompt_attention_mask_batch = kwargs.get("prompt_attention_mask_batch", None)
        guidance_scale = kwargs.get("guidance_scale", 5.0)
        stg_scale = kwargs.get("stg_scale", 1.0)
        rescaling_scale = kwargs.get("rescaling_scale", 1.0)
        num_videos = kwargs.get("num_videos", 1)
        fps = kwargs.get("fps", 25)
        image_cond_noise_scale = kwargs.get("image_cond_noise_scale", 0.0)
        generator = kwargs.get("generator", None)
        stochastic_sampling = kwargs.get("stochastic_sampling", False)
        cfg_star_rescale = kwargs.get("cfg_star_rescale", False)
        render_on_step = kwargs.get("render_on_step", False)
        render_on_step_callback = kwargs.get("render_on_step_callback", None)
        num_warmup_steps = kwargs.get("num_warmup_steps", 0)
        extra_step_kwargs = kwargs.get("extra_step_kwargs", {})
        pixel_coords = kwargs.get("pixel_coords", None)
        transformer_dtype = kwargs.get("transformer_dtype", None)
        skip_block_list = kwargs.get("skip_block_list", None)
        skip_layer_strategy = kwargs.get("skip_layer_strategy", None)
        init_latents = latents.clone()  # Used for image_cond_noise_update
        orig_conditioning_mask = conditioning_mask

        with self._progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                do_classifier_free_guidance = guidance_scale[i] > 1.0
                do_spatio_temporal_guidance = stg_scale[i] > 0
                do_rescaling = rescaling_scale[i] != 1.0

                num_conds = 1
                if do_classifier_free_guidance:
                    num_conds += 1
                if do_spatio_temporal_guidance:
                    num_conds += 1

                if do_classifier_free_guidance and do_spatio_temporal_guidance:
                    indices = slice(num_videos * 0, num_videos * 3)
                elif do_classifier_free_guidance:
                    indices = slice(num_videos * 0, num_videos * 2)
                elif do_spatio_temporal_guidance:
                    indices = slice(num_videos * 1, num_videos * 3)
                else:
                    indices = slice(num_videos * 1, num_videos * 2)

                skip_layer_mask: Optional[torch.Tensor] = None
                if do_spatio_temporal_guidance:
                    if skip_block_list is not None:
                        skip_layer_mask = self.transformer.create_skip_layer_mask(
                            num_videos, num_conds, num_conds - 1, skip_block_list[i]
                        )

                batch_pixel_coords = torch.cat([pixel_coords] * num_conds)
                conditioning_mask = orig_conditioning_mask

                if conditioning_mask is not None:
                    assert num_videos == 1
                    conditioning_mask = torch.cat([conditioning_mask] * num_conds)
                fractional_coords = batch_pixel_coords.to(torch.float32)
                fractional_coords[:, 0] = fractional_coords[:, 0] * (1.0 / fps)

                if conditioning_mask is not None and image_cond_noise_scale > 0.0:
                    latents = self.add_noise_to_image_conditioning_latents(
                        t,
                        init_latents,
                        latents,
                        image_cond_noise_scale,
                        orig_conditioning_mask,
                        generator,
                    )

                latent_model_input = (
                    torch.cat([latents] * num_conds) if num_conds > 1 else latents
                )
                latent_model_input = self.scheduler.scale_model_input(
                    latent_model_input, t
                )

                current_timestep = t
                if not torch.is_tensor(current_timestep):
                    # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
                    # This would be a good case for the `match` statement (Python 3.10+)
                    is_mps = latent_model_input.device.type == "mps"
                    if isinstance(current_timestep, float):
                        dtype = torch.float32 if is_mps else torch.float64
                    else:
                        dtype = torch.int32 if is_mps else torch.int64
                    current_timestep = torch.tensor(
                        [current_timestep],
                        dtype=dtype,
                        device=latent_model_input.device,
                    )
                elif len(current_timestep.shape) == 0:
                    current_timestep = current_timestep[None].to(
                        latent_model_input.device
                    )
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                current_timestep = current_timestep.expand(
                    latent_model_input.shape[0]
                ).unsqueeze(-1)

                if conditioning_mask is not None:
                    # Conditioning latents have an initial timestep and noising level of (1.0 - conditioning_mask)
                    # and will start to be denoised when the current timestep is lower than their conditioning timestep.
                    current_timestep = torch.min(
                        current_timestep, 1.0 - conditioning_mask
                    )

                noise_pred = self.transformer(
                    hidden_states=latent_model_input.to(transformer_dtype),
                    video_coords=fractional_coords,
                    encoder_hidden_states=prompt_embeds_batch[indices].to(
                        transformer_dtype
                    ),
                    encoder_attention_mask=prompt_attention_mask_batch[indices].to(
                        self.device
                    ),
                    timestep=current_timestep,
                    skip_layer_mask=skip_layer_mask,
                    skip_layer_strategy=skip_layer_strategy,
                    return_dict=False,
                )[0]

                init_noise_pred = noise_pred.clone()

                # perform guidance
                if do_spatio_temporal_guidance:
                    noise_pred_text, noise_pred_text_perturb = noise_pred.chunk(
                        num_conds
                    )[-2:]
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(num_conds)[:2]

                    if cfg_star_rescale:
                        # Rescales the unconditional noise prediction using the projection of the conditional prediction onto it:
                        # α = (⟨ε_text, ε_uncond⟩ / ||ε_uncond||²), then ε_uncond ← α * ε_uncond
                        # where ε_text is the conditional noise prediction and ε_uncond is the unconditional one.
                        positive_flat = noise_pred_text.view(num_videos, -1)
                        negative_flat = noise_pred_uncond.view(num_videos, -1)
                        dot_product = torch.sum(
                            positive_flat * negative_flat, dim=1, keepdim=True
                        )
                        squared_norm = (
                            torch.sum(negative_flat**2, dim=1, keepdim=True) + 1e-8
                        )
                        alpha = dot_product / squared_norm
                        noise_pred_uncond = alpha * noise_pred_uncond

                    noise_pred = noise_pred_uncond + guidance_scale[i] * (
                        noise_pred_text - noise_pred_uncond
                    )
                elif do_spatio_temporal_guidance:
                    noise_pred = noise_pred_text
                if do_spatio_temporal_guidance:
                    noise_pred = noise_pred + stg_scale[i] * (
                        noise_pred_text - noise_pred_text_perturb
                    )
                    if do_rescaling and stg_scale[i] > 0.0:
                        noise_pred_text_std = noise_pred_text.view(num_videos, -1).std(
                            dim=1, keepdim=True
                        )
                        noise_pred_std = noise_pred.view(num_videos, -1).std(
                            dim=1, keepdim=True
                        )

                        factor = noise_pred_text_std / noise_pred_std
                        factor = rescaling_scale[i] * factor + (1 - rescaling_scale[i])

                        noise_pred = noise_pred * factor.view(num_videos, 1, 1)

                current_timestep = current_timestep[:1]
                # learned sigma
                if (
                    self.transformer.config.out_channels // 2
                    == self.transformer.config.in_channels
                ):
                    noise_pred = noise_pred.chunk(2, dim=1)[0]

                # compute previous image: x_t -> x_t-1
                latents = self.denoising_step(
                    latents,
                    noise_pred,
                    current_timestep,
                    orig_conditioning_mask,
                    t,
                    extra_step_kwargs,
                    stochastic_sampling=stochastic_sampling,
                )

                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()

                if render_on_step and render_on_step_callback:
                    self._render_step(latents, render_on_step_callback)

        return latents

    def denoising_step(
        self,
        latents: torch.Tensor,
        noise_pred: torch.Tensor,
        current_timestep: torch.Tensor,
        conditioning_mask: torch.Tensor | None,
        t: float,
        extra_step_kwargs,
        t_eps=1e-6,
        stochastic_sampling=False,
    ):
        """
        Perform the denoising step for the required tokens, based on the current timestep and
        conditioning mask:
        Conditioning latents have an initial timestep and noising level of (1.0 - conditioning_mask)
        and will start to be denoised when the current timestep is equal or lower than their
        conditioning timestep.
        (hard-conditioning latents with conditioning_mask = 1.0 are never denoised)
        """
        # Denoise the latents using the scheduler
        denoised_latents = self.scheduler.step(
            noise_pred,
            t if current_timestep is None else current_timestep,
            latents,
            **extra_step_kwargs,
            return_dict=False,
            stochastic_sampling=stochastic_sampling,
        )[0]

        if conditioning_mask is None:
            return denoised_latents

        tokens_to_denoise_mask = (t - t_eps < (1.0 - conditioning_mask)).unsqueeze(-1)
        return torch.where(tokens_to_denoise_mask, denoised_latents, latents)

    @staticmethod
    def add_noise_to_image_conditioning_latents(
        t: float,
        init_latents: torch.Tensor,
        latents: torch.Tensor,
        noise_scale: float,
        conditioning_mask: torch.Tensor,
        generator,
        eps=1e-6,
    ):
        """
        Add timestep-dependent noise to the hard-conditioning latents. This helps with motion continuity, especially
        when conditioned on a single frame.
        """
        noise = randn_tensor(
            latents.shape,
            generator=generator,
            device=latents.device,
            dtype=latents.dtype,
        )
        # Add noise only to hard-conditioning latents (conditioning_mask = 1.0)
        need_to_noise = (conditioning_mask > 1.0 - eps).unsqueeze(-1)
        noised_latents = init_latents + noise_scale * noise * (t**2)
        latents = torch.where(need_to_noise, noised_latents, latents)
        return latents
