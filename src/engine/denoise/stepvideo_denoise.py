import torch
from src.utils.type_utils import EnumType


class DenoiseType(EnumType):
    BASE = "base"


class StepVideoDenoise:
    def __init__(self, denoise_type: DenoiseType = DenoiseType.BASE, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.denoise_type = denoise_type

    def denoise(self, *args, **kwargs) -> torch.Tensor:
        if self.denoise_type == DenoiseType.BASE:
            return self.base_denoise(*args, **kwargs)
        else:
            raise ValueError(f"Denoise type {self.denoise_type} not supported")

    def base_denoise(self, *args, **kwargs) -> torch.Tensor:
        timesteps = kwargs.get("timesteps", None)
        latents = kwargs.get("latents", None)
        transformer_dtype = kwargs.get("transformer_dtype", None)
        use_cfg_guidance = kwargs.get("use_cfg_guidance", True)
        render_on_step = kwargs.get("render_on_step", False)
        render_on_step_callback = kwargs.get("render_on_step_callback", None)
        scheduler = kwargs.get("scheduler", None)
        guidance_scale = kwargs.get("guidance_scale", 5.0)

        with self._progress_bar(
            len(timesteps), desc=f"Sampling {self.model_type}"
        ) as pbar:
            for t in timesteps:
                latent_model_input = (
                    torch.cat([latents] * 2) if use_cfg_guidance else latents
                )
                latent_model_input = latent_model_input.to(transformer_dtype)
                timestep = t.expand(latent_model_input.shape[0]).to(transformer_dtype)

                # Forward pass with both text and image conditioning
                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    return_dict=False,
                    **kwargs.get("transformer_kwargs", {}),
                )[0]

                if use_cfg_guidance:
                    noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )

                latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if render_on_step and render_on_step_callback:
                    self._render_step(latents, render_on_step_callback)
                pbar.update(1)

            self.logger.info("Denoising completed.")

        return latents
