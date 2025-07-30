import torch
import math
from src.utils.type_utils import EnumType
from src.utils.cache_utils import empty_cache

class DenoiseType(EnumType):
    BASE = "base"
    MOE = "moe"
    DIFFUSION_FORCING = "diffusion_forcing"
    MULTITALK = "multitalk"


class WanDenoise:
    def __init__(self, denoise_type: DenoiseType = DenoiseType.BASE, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.denoise_type = denoise_type

    def denoise(self, *args, **kwargs) -> torch.Tensor:
        if self.denoise_type == DenoiseType.BASE:
            return self.base_denoise(*args, **kwargs)
        elif self.denoise_type == DenoiseType.MOE:
            return self.moe_denoise(*args, **kwargs)
        elif self.denoise_type == DenoiseType.DIFFUSION_FORCING:
            return self.diffusion_forcing_denoise(*args, **kwargs)
        elif self.denoise_type == DenoiseType.MULTITALK:
            return self.multitalk_denoise(*args, **kwargs)
        else:
            raise ValueError(f"Denoise type {self.denoise_type} not supported")

    def moe_denoise(self, *args, **kwargs) -> torch.Tensor:
        timesteps = kwargs.get("timesteps", None)
        latents = kwargs.get("latents", None)
        latent_condition = kwargs.get("latent_condition", None)
        transformer_dtype = kwargs.get("transformer_dtype", None)
        use_cfg_guidance = kwargs.get("use_cfg_guidance", True)
        render_on_step = kwargs.get("render_on_step", False)
        render_on_step_callback = kwargs.get("render_on_step_callback", None)
        scheduler = kwargs.get("scheduler", None)
        guidance_scale = kwargs.get("guidance_scale", 5.0)
        boundary_timestep = kwargs.get("boundary_timestep", None)
        expand_timesteps = kwargs.get("expand_timesteps", False)
        mask = torch.ones(latents.shape, dtype=torch.float32, device=self.device)
        
        with self._progress_bar(
            len(timesteps), desc=f"Sampling MOE"
        ) as pbar:
            for i, t in enumerate(timesteps):
                if expand_timesteps:
                    # seq_len: num_latent_frames * latent_height//2 * latent_width//2
                    temp_ts = (mask[0][0][:, ::2, ::2] * t).flatten()
                    # batch_size, seq_len
                    timestep = temp_ts.unsqueeze(0).expand(latents.shape[0], -1)
                else:
                    timestep = t.expand(latents.shape[0])
                
                if latent_condition is not None:
                    latent_model_input = torch.cat(
                        [latents, latent_condition], dim=1
                    ).to(transformer_dtype)
                else:
                    latent_model_input = latents.to(transformer_dtype)
                
                
                if boundary_timestep is None or t >= boundary_timestep:
                    if hasattr(self, "transformer_2") and self.transformer_2:
                        self._offload(self.transformer_2)
                        setattr(self, "transformer_2", None)
                        empty_cache()
                        
                    if not self.transformer:
                        self.load_component_by_name("transformer")
                        self.to_device(self.transformer)
                    
                    transformer = self.transformer
                    if isinstance(guidance_scale, list):
                        guidance_scale = guidance_scale[1]
                else:
                    if self.transformer:
                        self._offload(self.transformer)
                        setattr(self, "transformer", None)
                        empty_cache()
                    
                    if not hasattr(self, "transformer_2") or not self.transformer_2:  
                        self.load_component_by_name("transformer_2")
                        self.to_device(self.transformer_2)
                    
                    transformer = self.transformer_2
                    if isinstance(guidance_scale, list):
                        guidance_scale = guidance_scale[0]
                    # Standard denoising
                noise_pred = transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    return_dict=False,
                    **kwargs.get("transformer_kwargs", {}),
                )[0]
                if use_cfg_guidance and kwargs.get(
                    "unconditional_transformer_kwargs", None
                ):
                    uncond_noise_pred = transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep,
                        return_dict=False,
                        **kwargs.get("unconditional_transformer_kwargs", {}),
                    )[0]
                    noise_pred = uncond_noise_pred + guidance_scale * (
                        noise_pred - uncond_noise_pred
                    )
                latents = scheduler.step(noise_pred, t, latents, return_dict=False)[
                    0
                ]

                if render_on_step and render_on_step_callback:
                    self._render_step(latents, render_on_step_callback)
                pbar.update(1)

            self.logger.info("Denoising completed.")

        return latents
    
    
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
        
        if not self.transformer:
            self.load_component_by_type("transformer")
        self.to_device(self.transformer)    
        
        model_type_str = getattr(self, "model_type", "WAN")
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
                    # Standard denoising
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

                    latents = scheduler.step(noise_pred, t, latents, return_dict=False)[
                        0
                    ]

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
            total=len(step_matrix),
            desc=f"Sampling {getattr(self, 'model_type', 'WAN')}",
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
    
    def multitalk_denoise(self, *args, **kwargs) -> torch.Tensor:
        latents = kwargs.get("latents", None)
        latent_condition = kwargs.get("latent_condition", None)
        transformer_dtype = kwargs.get("transformer_dtype", None)
        use_cfg_guidance = kwargs.get("use_cfg_guidance", True)
        is_first_clip = kwargs.get("is_first_clip", True)
        latent_motion_frames = kwargs.get("latent_motion_frames", None)
        render_on_step = kwargs.get("render_on_step", False)
        render_on_step_callback = kwargs.get("render_on_step_callback", None)
        scheduler = kwargs.get("scheduler", None)
        guidance_scale = kwargs.get("guidance_scale", 5.0)
        audio_guidance_scale = kwargs.get("audio_guidance_scale", None)
        timesteps = kwargs.get("timesteps", None)
        prompt_embeds = kwargs.get("prompt_embeds", None)
        image_embeds = kwargs.get("image_embeds", None)
        audio_embeds = kwargs.get("audio_embeds", None)
        ref_target_masks = kwargs.get("ref_target_masks", None)
        human_num = kwargs.get("human_num", None)
        negative_prompt_embeds = kwargs.get("negative_prompt_embeds", None)
        attention_kwargs = kwargs.get("attention_kwargs", {})

        with self._progress_bar(
            len(timesteps), desc=f"Sampling MULTITALK"
        ) as pbar:
            for i, t in enumerate(timesteps):
                latent_model_input = torch.cat([latents, latent_condition], dim=1).to(transformer_dtype)
                
                noise_pred_cond = self.transformer(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    encoder_hidden_states_image=image_embeds,
                    encoder_hidden_states_audio=audio_embeds,
                    ref_target_masks=ref_target_masks,
                    human_num=human_num,
                    return_dict=False,
                    **attention_kwargs,
                )[0]
                
                if math.isclose(guidance_scale, 1.0):
                    noise_pred_drop_audio = self.transformer(
                        latent_model_input,
                        t,
                        encoder_hidden_states=prompt_embeds,
                        encoder_hidden_states_image=image_embeds,
                        encoder_hidden_states_audio=torch.zeros_like(audio_embeds)[-1:],
                        ref_target_masks=ref_target_masks,
                        human_num=human_num,
                        return_dict=False,
                        **attention_kwargs,
                    )[0]
                else:
                    noise_pred_drop_text = self.transformer(
                        latent_model_input,
                        t,
                        encoder_hidden_states=negative_prompt_embeds,
                        encoder_hidden_states_image=image_embeds,
                        encoder_hidden_states_audio=audio_embeds,
                        ref_target_masks=ref_target_masks,
                        human_num=human_num,
                        return_dict=False,
                        **attention_kwargs,
                    )[0]
                    
                    noise_pred_uncond = self.transformer(
                        latent_model_input,
                        t,
                        encoder_hidden_states=negative_prompt_embeds,
                        encoder_hidden_states_image=image_embeds,
                        encoder_hidden_states_audio=torch.zeros_like(audio_embeds)[-1:],
                        ref_target_masks=ref_target_masks,
                        human_num=human_num,
                        return_dict=False,
                        **attention_kwargs,
                    )[0]
                    
                if math.isclose(guidance_scale, 1.0):
                            noise_pred = noise_pred_drop_audio + audio_guidance_scale* (noise_pred_cond - noise_pred_drop_audio)  
                else:
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_cond - noise_pred_drop_text) + \
                        audio_guidance_scale * (noise_pred_drop_text - noise_pred_uncond)  
                noise_pred = -noise_pred  
                
                latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                
                if not is_first_clip:
                    latent_motion_frames = latent_motion_frames.to(latents.dtype).to(self.device)
                    motion_add_noise = torch.randn_like(latent_motion_frames).contiguous()
                    add_latent = scheduler.add_noise(latent_motion_frames, motion_add_noise, timesteps[i+1])
                    _, T_m, _, _ = add_latent.shape
                    latents[:, :T_m] = add_latent

                if render_on_step and render_on_step_callback:
                    self._render_step(latents, render_on_step_callback)
                pbar.update(1)

            self.logger.info("Denoising completed.")

        return latents
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                