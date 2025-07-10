import torch
from enum import Enum


class DenoiseType(Enum):
    BASE = "base"
    DIFFUSION_FORCING = "diffusion_forcing"


class WanDenoise:
    def __init__(self, denoise_type: DenoiseType = DenoiseType.BASE, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.denoise_type = denoise_type

    def denoise(self, *args, **kwargs) -> torch.Tensor:
        if self.denoise_type == DenoiseType.BASE:
            return self.base_denoise(*args, **kwargs)
        elif self.denoise_type == DenoiseType.DIFFUSION_FORCING:
            return self.diffusion_forcing_denoise(*args, **kwargs)
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

    def diffusion_forcing_denoise(self, *args, **kwargs) -> torch.Tensor:
        latents = kwargs.get("latents", None)
        transformer_dtype = kwargs.get("transformer_dtype", None)
        use_cfg_guidance = kwargs.get("use_cfg_guidance", True)
        render_on_step = kwargs.get("render_on_step", False)
        render_on_step_callback = kwargs.get("render_on_step_callback", None)
        guidance_scale = kwargs.get("guidance_scale", 5.0)
        fps_embeds = kwargs.get("fps_embeds", None)
        prompt_embeds = kwargs.get("prompt_embeds", None)
        negative_prompt_embeds = kwargs.get("negative_prompt_embeds", None)
        attention_kwargs = kwargs.get("attention_kwargs", {})
        addnoise_condition = kwargs.get("addnoise_condition", 0)
        encoded_image_length = kwargs.get("encoded_image_length", None)
        step_matrix = kwargs.get("step_matrix", None)
        step_update_mask = kwargs.get("step_update_mask", None)
        valid_interval = kwargs.get("valid_interval", None)
        schedulers_counter = kwargs.get("schedulers_counter", None)
        schedulers = kwargs.get("schedulers", None)

        with self._progress_bar(
            total=len(step_matrix), desc=f"Sampling {self.model_type}"
        ) as pbar:
            for i, timestep_i in enumerate(step_matrix):
                update_mask_i = step_update_mask[i]
                valid_interval_i = valid_interval[i]

                valid_interval_start, valid_interval_end = valid_interval_i
                timestep = timestep_i[
                    None, valid_interval_start:valid_interval_end
                ].clone()
                latent_model_input = (
                    latents[:, :, valid_interval_start:valid_interval_end, :, :]
                    .clone()
                    .to(self.device, dtype=transformer_dtype)
                )

                if (
                    addnoise_condition > 0
                    and valid_interval_start < encoded_image_length
                ):
                    noise_factor = 0.001 * addnoise_condition
                    timestep_for_noised_condition = addnoise_condition
                    latent_model_input[
                        0, :, valid_interval_start:encoded_image_length, :, :
                    ] = (
                        latent_model_input[
                            0, :, valid_interval_start:encoded_image_length, :, :
                        ]
                        * (1.0 - noise_factor)
                        + torch.randn_like(
                            latent_model_input[
                                0, :, valid_interval_start:encoded_image_length, :, :
                            ]
                        )
                        * noise_factor
                    )
                    timestep[:, valid_interval_start:encoded_image_length] = (
                        timestep_for_noised_condition
                    )

                noise_pred = self.transformer(
                    latent_model_input,
                    timestep,
                    encoder_hidden_states_fps=fps_embeds,
                    encoder_hidden_states=prompt_embeds,
                    return_dict=False,
                    **attention_kwargs,
                )[0]

                if use_cfg_guidance:
                    uncond_noise_pred = self.transformer(
                        latent_model_input,
                        timestep,
                        encoder_hidden_states_fps=fps_embeds,
                        encoder_hidden_states=negative_prompt_embeds,
                        return_dict=False,
                        **attention_kwargs,
                    )[0]
                    noise_pred = uncond_noise_pred + guidance_scale * (
                        noise_pred - uncond_noise_pred
                    )

                for idx in range(valid_interval_start, valid_interval_end):
                    if update_mask_i[idx].item():
                        latents[0, :, idx] = schedulers[idx].step(
                            noise_pred[0, :, idx - valid_interval_start],
                            timestep_i[idx],
                            latents[0, :, idx],
                            return_dict=False,
                        )[0]
                        schedulers_counter[idx] += 1

                if render_on_step and render_on_step_callback:
                    self._render_step(latents, render_on_step_callback)

                pbar.update(1)

        return latents
