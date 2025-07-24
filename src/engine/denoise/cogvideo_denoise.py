import torch
import math
from enum import Enum
from diffusers.schedulers import CogVideoXDPMScheduler
from contextlib import nullcontext


class CogVideoDenoiseType(Enum):
    T2V = "t2v"
    I2V = "i2v"
    V2V = "v2v"
    FUN = "fun"


class CogVideoDenoise:
    def __init__(
        self,
        denoise_type: CogVideoDenoiseType = CogVideoDenoiseType.T2V,
        *args,
        **kwargs,
    ):
        self.denoise_type = denoise_type

    def denoise(self, *args, **kwargs) -> torch.Tensor:
        """Unified denoising loop for all CogVideo modes"""
        latents = kwargs.get("latents", None)
        timesteps = kwargs.get("timesteps", None)
        scheduler = kwargs.get("scheduler", None)
        guidance_scale = kwargs.get("guidance_scale", 6.0)
        use_dynamic_cfg = kwargs.get("use_dynamic_cfg", False)
        do_classifier_free_guidance = kwargs.get("do_classifier_free_guidance", False)
        noise_pred_kwargs = kwargs.get("noise_pred_kwargs", {})
        render_on_step = kwargs.get("render_on_step", False)
        render_on_step_callback = kwargs.get("render_on_step_callback", None)
        num_inference_steps = kwargs.get("num_inference_steps", 50)
        transformer_dtype = kwargs.get("transformer_dtype", None)
        extra_step_kwargs = kwargs.get("extra_step_kwargs", {})

        # Mode-specific inputs
        image_latents = kwargs.get("image_latents", None)

        num_warmup_steps = max(
            len(timesteps) - num_inference_steps * scheduler.order, 0
        )

        # Get mode-specific description
        mode_desc = {
            CogVideoDenoiseType.T2V: "T2V",
            CogVideoDenoiseType.I2V: "I2V",
            CogVideoDenoiseType.V2V: "V2V",
            CogVideoDenoiseType.FUN: "Fun",
        }.get(self.denoise_type, "CogVideo")

        with self._progress_bar(
            total=num_inference_steps, desc=f"Denoising {mode_desc}"
        ) as pbar:
            old_pred_original_sample = None
            for i, t in enumerate(timesteps):
                # Expand latents if doing classifier free guidance
                latent_model_input = (
                    torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                )
                latent_model_input = scheduler.scale_model_input(
                    latent_model_input, t
                ).to(transformer_dtype)

                # Mode-specific input preparation
                if (
                    self.denoise_type == CogVideoDenoiseType.I2V
                    and image_latents is not None
                ):
                    # Concatenate with image latents for I2V
                    latent_image_input = (
                        torch.cat([image_latents] * 2)
                        if do_classifier_free_guidance
                        else image_latents
                    )
                    latent_model_input = torch.cat(
                        [latent_model_input, latent_image_input], dim=2
                    ).to(transformer_dtype)

                # Broadcast timestep to batch dimension
                timestep = t.expand(latent_model_input.shape[0])

                # Predict noise
                if hasattr(self.transformer, "cache_context"):
                    cache_context = self.transformer.cache_context(
                        "cond_uncond" if do_classifier_free_guidance else None
                    )
                else:
                    cache_context = nullcontext()

                with cache_context:
                    noise_pred = self.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep,
                        return_dict=False,
                        **noise_pred_kwargs,
                    )[0]

                noise_pred = noise_pred.float()

                # Perform guidance
                if use_dynamic_cfg:
                    # Dynamic CFG scaling based on timestep
                    dynamic_guidance_scale = 1 + guidance_scale * (
                        (
                            1
                            - math.cos(
                                math.pi
                                * (
                                    (num_inference_steps - t.item())
                                    / num_inference_steps
                                )
                                ** 5.0
                            )
                        )
                        / 2
                    )
                else:
                    dynamic_guidance_scale = guidance_scale

                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + dynamic_guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )

                # Scheduler step - handle different scheduler types
                if not isinstance(self.scheduler, CogVideoXDPMScheduler):
                    latents = self.scheduler.step(
                        noise_pred, t, latents, **extra_step_kwargs, return_dict=False
                    )[0]
                else:
                    latents, old_pred_original_sample = self.scheduler.step(
                        noise_pred,
                        old_pred_original_sample,
                        t,
                        timesteps[i - 1] if i > 0 else None,
                        latents,
                        **extra_step_kwargs,
                        return_dict=False,
                    )

                latents = latents.to(transformer_dtype)

                if render_on_step and render_on_step_callback:
                    self._render_step(latents, render_on_step_callback)

                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % scheduler.order == 0
                ):
                    pbar.update(1)

        self.logger.info(f"{mode_desc} denoising completed.")
        return latents
