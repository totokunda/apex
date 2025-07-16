import torch
from enum import Enum


class DenoiseType(Enum):
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
        latent_condition = kwargs.get("latent_condition", None)
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
                timestep = t.expand(latents.shape[0])
                if latent_condition is not None:
                    latent_model_input = torch.cat(
                        [latents, latent_condition], dim=1
                    ).to(transformer_dtype)
                else:
                    latent_model_input = latents.to(transformer_dtype)

                # Forward pass with both text and image conditioning
                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    return_dict=False,
                    **kwargs.get("transformer_kwargs", {}),
                )[0]

                if use_cfg_guidance and kwargs.get(
                    "unconditional_transformer_kwargs", None
                ):
                    uncond_noise_pred = self.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep,
                        return_dict=False,
                        **kwargs.get("unconditional_transformer_kwargs", {}),
                    )[0]

                    noise_pred = uncond_noise_pred + guidance_scale * (
                        noise_pred - uncond_noise_pred
                    )

                latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if render_on_step and render_on_step_callback:
                    self._render_step(latents, render_on_step_callback)
                pbar.update(1)

            self.logger.info("Denoising completed.")

        return latents
