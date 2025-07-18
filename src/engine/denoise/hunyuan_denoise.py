import torch
from enum import Enum

class DenoiseType(Enum):
    BASE = "base"
    HYAVATAR = "hyavatar"


class HunyuanDenoise:
    def __init__(self, denoise_type: DenoiseType = DenoiseType.T2V, *args, **kwargs):
        self.denoise_type = denoise_type

    def denoise(self, *args, **kwargs) -> torch.Tensor:
        if self.denoise_type == DenoiseType.BASE:
            return self.base_denoise(*args, **kwargs)
        elif self.denoise_type == DenoiseType.HYAVATAR:
            return self.hyavatar_denoise(*args, **kwargs)

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

    def hyavatar_denoise(self, *args, **kwargs) -> torch.Tensor:
        infer_length = kwargs.get("infer_length", None)
        latents_all = kwargs.get("latents_all", None)
        audio_prompts = kwargs.get("audio_prompts", None)
        uncond_audio_prompts = kwargs.get("uncond_audio_prompts", None)
        face_masks = kwargs.get("face_masks", None)
        negative_prompt_embeds = kwargs.get("negative_prompt_embeds", None)
        prompt_embeds = kwargs.get("prompt_embeds", None)
        negative_prompt_attention_mask = kwargs.get("negative_prompt_attention_mask", None)
        prompt_attention_mask = kwargs.get("prompt_attention_mask", None)
        pooled_prompt_embeds = kwargs.get("pooled_prompt_embeds", None)
        negative_pooled_prompt_embeds = kwargs.get("negative_pooled_prompt_embeds", None)
        ref_latents = kwargs.get("ref_latents", None)
        uncond_ref_latents = kwargs.get("uncond_ref_latents", None)
        timesteps = kwargs.get("timesteps", None)
        num_inference_steps = kwargs.get("num_inference_steps", 50)
        guidance_scale = kwargs.get("guidance_scale", 6.0)
        motion_exp = kwargs.get("motion_exp", None)
        motion_pose = kwargs.get("motion_pose", None)
        fps_tensor = kwargs.get("fps_tensor", None)
        freqs_cis = kwargs.get("freqs_cis", None)
        num_videos = kwargs.get("num_videos", 1)
        
        
        frames_per_batch = 33  # From the reference implementation
        shift = 0
        shift_offset = 10  # From the reference implementation

        if infer_length <= frames_per_batch:
            shift_offset = 0

        no_cache_steps = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14] + list(range(15, 42, 5)) + [41, 42, 43, 44, 45, 46, 47, 48, 49]
        cache_tensor = {}
        with self._progress_bar(total=num_inference_steps, desc="Denoising HyAvatar") as progress_bar:
            for i, t in enumerate(timesteps):

                pred_latents = torch.zeros_like(latents_all, dtype=latents_all.dtype)
                counter = torch.zeros_like(latents_all, dtype=latents_all.dtype)

                for index_start in range(0, infer_length, frames_per_batch):
                    index_start = index_start - shift

                    idx_list = [
                        ii % infer_length
                        for ii in range(index_start, index_start + frames_per_batch)
                    ]
                    latents = latents_all[:, :, idx_list].clone()

                    idx_list_audio = [
                        ii % audio_prompts.shape[1]
                        for ii in range(
                            index_start * 4, (index_start + frames_per_batch) * 4
                        )
                    ]

                    # Ensure audio prompt list is not out of bounds
                    if max(idx_list_audio) >= audio_prompts.shape[1]:
                        idx_list_audio = [
                            min(i, audio_prompts.shape[1] - 1) for i in idx_list_audio
                        ]

                    current_audio_prompts = audio_prompts[:, idx_list_audio].clone()
                    current_uncond_audio_prompts = uncond_audio_prompts[
                        :, idx_list_audio
                    ].clone()

                    # Classifier-Free Guidance setup
                    latent_model_input = torch.cat([latents] * 2)
                    latent_model_input = self.scheduler.scale_model_input(
                        latent_model_input, t
                    )

                    # Dynamic Guidance
                    if i < 10:
                        current_guidance_scale = (1 - i / len(timesteps)) * (
                            guidance_scale - 2
                        ) + 2
                        current_face_masks = face_masks * 0.6
                        text_embeds_input = torch.cat(
                            [negative_prompt_embeds, prompt_embeds]
                        )
                        text_mask_input = torch.cat(
                            [negative_prompt_attention_mask, prompt_attention_mask]
                        )
                        pooled_embeds_input = torch.cat(
                            [negative_pooled_prompt_embeds, pooled_prompt_embeds]
                        )
                    else:
                        current_guidance_scale = (1 - i / len(timesteps)) * (
                            6.5 - 3.5
                        ) + 3.5
                        current_face_masks = face_masks
                        text_embeds_input = torch.cat(
                            [prompt_embeds, prompt_embeds]
                        )  # Use conditional prompts for both
                        text_mask_input = torch.cat(
                            [prompt_attention_mask, prompt_attention_mask]
                        )
                        pooled_embeds_input = torch.cat(
                            [pooled_prompt_embeds, pooled_prompt_embeds]
                        )

                    # Concatenate inputs for CFG
                    audio_prompts_input = torch.cat(
                        [current_uncond_audio_prompts, current_audio_prompts]
                    )
                    face_masks_input = torch.cat([current_face_masks] * 2)
                    ref_latents_input = torch.cat(
                        [uncond_ref_latents, ref_latents] * num_videos
                    )
                    motion_exp_input = torch.cat([motion_exp] * 2)
                    motion_pose_input = torch.cat([motion_pose] * 2)
                    fps_input = torch.cat([fps_tensor] * 2)

                    
                    latent_input_len = (latent_model_input.shape[-1] // 2)  * (latent_model_input.shape[-2] // 2) * latent_model_input.shape[-3]
                    latent_ref_len = (latent_model_input.shape[-1] // 2)  * (latent_model_input.shape[-2] // 2) * (latent_model_input.shape[-3]+1) 
                    
                    if i in no_cache_steps:
                        use_cache = False

                        noise_pred = self.transformer(
                            hidden_states=latent_model_input,
                            timestep=t,
                            encoder_hidden_states=text_embeds_input,
                            encoder_attention_mask=text_mask_input,
                            pooled_projections=pooled_embeds_input,
                            ref_latents=ref_latents_input,
                            encoder_hidden_states_face_mask=face_masks_input,
                            encoder_hidden_states_audio=audio_prompts_input,
                            encoder_hidden_states_motion=motion_exp_input,
                            encoder_hidden_states_pose=motion_pose_input,
                            encoder_hidden_states_fps=fps_input,
                            freqs_cos=freqs_cis[0],
                            freqs_sin=freqs_cis[1],
                            use_cache=use_cache,
                            return_dict=False,
                        )[0]

                        if not cache_tensor:
                            cache_tensor = {
                                "reference_latent": torch.zeros([latent_model_input.shape[0], latents_all.shape[-3], (latent_model_input.shape[-1] // 2)  * (latent_model_input.shape[-2] // 2), 3072]).to(self.transformer.latent_cache.dtype).to(latent_model_input.device).clone(),
                                "input_latent": torch.zeros([latent_model_input.shape[0], latents_all.shape[-3], (latent_model_input.shape[-1] // 2)  * (latent_model_input.shape[-2] // 2), 3072]).to(self.transformer.latent_cache.dtype).to(latent_model_input.device).clone(),
                                "text_embeds": torch.zeros([latent_model_input.shape[0], latents_all.shape[-3], text_embeds_input.shape[1], 3072]).to(self.transformer.latent_cache.dtype).to(latent_model_input.device).clone(),
                            }

                        cache_tensor["reference_latent"][:, idx_list] = self.transformer.latent_cache[:, :latent_ref_len-latent_input_len].reshape(latent_model_input.shape[0], 1, -1, 3072).repeat(1, len(idx_list), 1, 1).to(latent_model_input.device)
                        cache_tensor["input_latent"][:, idx_list] = self.transformer.latent_cache[:, latent_ref_len-latent_input_len:latent_ref_len].reshape(latent_model_input.shape[0], len(idx_list), -1, 3072).to(latent_model_input.device)
                        cache_tensor["text_embeds"][:, idx_list] = self.transformer.latent_cache[:, latent_ref_len:].unsqueeze(1).repeat(1, len(idx_list), 1, 1).to(latent_model_input.device)
                    else:
                        use_cache = True
                        self.transformer.latent_cache[:, :latent_ref_len-latent_input_len] = cache_tensor["reference_latent"][:, idx_list][:, 0].clone()
                        self.transformer.latent_cache[:, latent_ref_len-latent_input_len:latent_ref_len] = cache_tensor["input_latent"][:, idx_list].reshape(-1, latent_input_len, 3072).clone()
                        self.transformer.latent_cache[:, latent_ref_len:] = cache_tensor["text_embeds"][:, idx_list][:, 0].clone()
                    
                    # Perform guidance
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + current_guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )

                    # Scheduler step
                    latents_step = self.scheduler.step(
                        noise_pred, t, latents, return_dict=False
                    )[0]

                    for latent_idx, master_idx in enumerate(idx_list):
                        pred_latents[:, :, master_idx] += latents_step[:, :, latent_idx]
                        counter[:, :, master_idx] += 1

                shift += shift_offset
                shift = shift % frames_per_batch

                # Average the predictions from the sliding windows
                latents_all = pred_latents / counter.clamp(min=1)

                progress_bar.update()
        
        self.logger.info("Denoising completed.")
        return latents_all