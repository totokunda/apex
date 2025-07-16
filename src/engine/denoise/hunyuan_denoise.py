import torch
from enum import Enum
from typing import Optional


class DenoiseType(Enum):
    BASE = "base"


class HunyuanDenoise:
    def __init__(self, denoise_type: DenoiseType = DenoiseType.T2V, *args, **kwargs):
        self.denoise_type = denoise_type

    def denoise(self, *args, **kwargs) -> torch.Tensor:
        if self.denoise_type == DenoiseType.BASE:
            return self.base_denoise(*args, **kwargs)

    def base_denoise(self, *args, **kwargs) -> torch.Tensor:
        timesteps = kwargs.get("timesteps", None)
        latents = kwargs.get("latents", None)
        transformer_dtype = kwargs.get("transformer_dtype", None)
        use_true_cfg_guidance = kwargs.get("use_true_cfg_guidance", False)
        render_on_step = kwargs.get("render_on_step", False)
        render_on_step_callback = kwargs.get("render_on_step_callback", None)
        scheduler = kwargs.get("scheduler", None)
        true_guidance_scale = kwargs.get("true_guidance_scale", 1.0)
        num_inference_steps = kwargs.get("num_inference_steps", 50)
        image_condition_type = kwargs.get("image_condition_type", None)
        image_latents = kwargs.get("image_latents", None)
        mask = kwargs.get("mask", None)
        noise_pred_kwargs = kwargs.get("noise_pred_kwargs", {})
        unconditional_noise_pred_kwargs = kwargs.get(
            "unconditional_noise_pred_kwargs", {}
        )

        with self._progress_bar(
            total=num_inference_steps, desc="Denoising T2V"
        ) as pbar:
            for i, t in enumerate(timesteps):
                if image_condition_type == "latent_concat":
                    latent_model_input = torch.cat(
                        [latents, image_latents, mask], dim=1
                    ).to(transformer_dtype)
                elif image_condition_type == "token_replace":
                    latent_model_input = torch.cat(
                        [image_latents, latents[:, :, 1:]], dim=2
                    ).to(transformer_dtype)
                else:
                    latent_model_input = latents.to(transformer_dtype)
                timestep = t.expand(latents.shape[0]).to(latents.dtype)

                # Conditional forward pass
                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    return_dict=False,
                    **noise_pred_kwargs,
                )[0]

                # Unconditional forward pass for CFG
                if use_true_cfg_guidance:
                    neg_noise_pred = self.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep,
                        return_dict=False,
                        **unconditional_noise_pred_kwargs,
                    )[0]
                    noise_pred = neg_noise_pred + true_guidance_scale * (
                        noise_pred - neg_noise_pred
                    )

                # Scheduler step
                if image_condition_type == "latent_concat":
                    latents = scheduler.step(noise_pred, t, latents, return_dict=False)[
                        0
                    ]
                elif image_condition_type == "token_replace":
                    latents_step = scheduler.step(
                        noise_pred[:, :, 1:], t, latents[:, :, 1:], return_dict=False
                    )[0]
                    latents = torch.cat([image_latents, latents_step], dim=2)
                else:
                    latents = scheduler.step(noise_pred, t, latents, return_dict=False)[
                        0
                    ]

                if render_on_step and render_on_step_callback:
                    self._render_step(latents, render_on_step_callback)

                pbar.update(1)

        self.logger.info("Denoising completed.")

        return latents
