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
        audio_guidance_scale = kwargs.get("audio_guidance_scale", None)
        shift = kwargs.get("shift", None)
        use_timestep_transform = kwargs.get("use_timestep_transform", False)

        # Apply timestep transformation if needed
        if use_timestep_transform and shift is not None:
            timesteps = [self._transform_timestep(t, shift) for t in timesteps]

        model_type_str = getattr(self, 'model_type', 'WAN')
        with self._progress_bar(
            len(timesteps), desc=f"Sampling {model_type_str}"
        ) as pbar:
            for i, t in enumerate(timesteps):
                timestep = t.expand(latents.shape[0])
                if latent_condition is not None:
                    latent_model_input = torch.cat(
                        [latents, latent_condition], dim=1
                    ).to(transformer_dtype)
                else:
                    latent_model_input = latents.to(transformer_dtype)

                # MultiTalk-specific triple-branch CFG
                if audio_guidance_scale is not None and use_cfg_guidance:
                    # Get transformer kwargs for different branches
                    transformer_kwargs = kwargs.get("transformer_kwargs", {})
                    unconditional_kwargs = kwargs.get("unconditional_transformer_kwargs", {})
                    
                    # Prepare inputs for all three branches
                    batch_latent_input = torch.cat([latent_model_input] * 3)  # cond, drop_text, uncond
                    batch_timestep = torch.cat([timestep] * 3)
                    
                    # Prepare conditional kwargs
                    cond_kwargs = transformer_kwargs.copy()
                    
                    # Prepare text-dropped kwargs (keep audio)
                    drop_text_kwargs = transformer_kwargs.copy()
                    if "encoder_hidden_states" in drop_text_kwargs and unconditional_kwargs.get("encoder_hidden_states") is not None:
                        drop_text_kwargs["encoder_hidden_states"] = unconditional_kwargs["encoder_hidden_states"]
                    
                    # Prepare unconditional kwargs (no text, no audio)
                    uncond_kwargs = unconditional_kwargs.copy()
                    
                    # Combine all kwargs for batch processing
                    combined_kwargs = {}
                    for key in cond_kwargs.keys():
                        if key == "attention_kwargs":
                            combined_kwargs[key] = cond_kwargs[key]
                            continue
                            
                        cond_val = cond_kwargs.get(key)
                        drop_text_val = drop_text_kwargs.get(key)
                        uncond_val = uncond_kwargs.get(key)
                        
                        if cond_val is not None:
                            if drop_text_val is not None and uncond_val is not None:
                                combined_kwargs[key] = torch.cat([cond_val, drop_text_val, uncond_val], dim=0)
                            else:
                                combined_kwargs[key] = torch.cat([cond_val] * 3, dim=0)
                    
                    # Forward pass for all branches
                    try:
                        noise_pred = self.transformer(
                            hidden_states=batch_latent_input,
                            timestep=batch_timestep,
                            return_dict=False,
                            **combined_kwargs,
                        )[0]
                        
                        # Split predictions
                        noise_pred_cond, noise_pred_drop_text, noise_pred_uncond = noise_pred.chunk(3)
                        
                        # Apply MultiTalk CFG
                        import math
                        if math.isclose(guidance_scale, 1.0):
                            # Only audio guidance
                            noise_pred = noise_pred_drop_text + audio_guidance_scale * (noise_pred_cond - noise_pred_drop_text)
                        else:
                            # Both text and audio guidance
                            noise_pred = (
                                noise_pred_uncond + 
                                guidance_scale * (noise_pred_cond - noise_pred_drop_text) +
                                audio_guidance_scale * (noise_pred_drop_text - noise_pred_uncond)
                            )
                        
                        # Apply flow matching direction
                        noise_pred = -noise_pred
                        
                    except Exception as e:
                        print(f"Warning: MultiTalk CFG failed, falling back to standard CFG: {e}")
                        # Fallback to standard CFG
                        noise_pred = self.transformer(
                            hidden_states=latent_model_input,
                            timestep=timestep,
                            return_dict=False,
                            **transformer_kwargs,
                        )[0]
                        
                        if unconditional_kwargs:
                            uncond_noise_pred = self.transformer(
                                hidden_states=latent_model_input,
                                timestep=timestep,
                                return_dict=False,
                                **unconditional_kwargs,
                            )[0]
                            noise_pred = uncond_noise_pred + guidance_scale * (noise_pred - uncond_noise_pred)
                    
                    # Update latents for flow matching
                    if use_timestep_transform and shift is not None:
                        dt = timesteps[i] - (timesteps[i + 1] if i < len(timesteps) - 1 else 0)
                        dt = dt / 1000.0  # Normalize to [0, 1]
                        latents = latents + noise_pred * dt
                    else:
                        latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                
                else:
                    # Standard denoising
                    noise_pred = self.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep,
                        return_dict=False,
                        **kwargs.get("transformer_kwargs", {}),
                    )[0]

                    if use_cfg_guidance and kwargs.get("unconditional_transformer_kwargs", None):
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

    def _transform_timestep(self, t: torch.Tensor, shift: float = 5.0, num_timesteps: int = 1000) -> torch.Tensor:
        """Apply timestep transformation for better generation dynamics."""
        t_normalized = t.float() / num_timesteps
        new_t = shift * t_normalized / (1 + (shift - 1) * t_normalized)
        return new_t * num_timesteps

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
            total=len(step_matrix), desc=f"Sampling {getattr(self, 'model_type', 'WAN')}"
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
