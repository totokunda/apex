import torch
from src.utils.type import EnumType


class DenoiseType(EnumType):
    BASE = "base"
    AVATAR = "avatar"


class HunyuanDenoise:
    def __init__(self, denoise_type: DenoiseType = DenoiseType.BASE, *args, **kwargs):
        self.denoise_type = denoise_type

    def denoise(self, *args, **kwargs) -> torch.Tensor:
        if self.denoise_type == DenoiseType.BASE:
            return self.base_denoise(*args, **kwargs)
        elif self.denoise_type == DenoiseType.AVATAR:
            return self.avatar_denoise(*args, **kwargs)

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
            total=num_inference_steps, desc=f"Denoising {self.denoise_type}"
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
                with self.transformer.cache_context("cond"):
                    noise_pred = self.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep,
                        return_dict=False,
                        **noise_pred_kwargs,
                    )[0]

                # Unconditional forward pass for CFG
                if use_true_cfg_guidance:
                    with self.transformer.cache_context("uncond"):
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

    def avatar_denoise(self, *args, **kwargs) -> torch.Tensor:
        infer_length = kwargs.get("infer_length", None)
        latents_all = kwargs.get("latents_all", None)
        audio_prompts_all = kwargs.get("audio_prompts_all", None)
        uncond_audio_prompts = kwargs.get("uncond_audio_prompts", None)
        face_masks = kwargs.get("face_masks", None)
        negative_prompt_embeds = kwargs.get("negative_prompt_embeds", None)
        prompt_embeds = kwargs.get("prompt_embeds", None)
        negative_prompt_attention_mask = kwargs.get(
            "negative_prompt_attention_mask", None
        )
        prompt_attention_mask = kwargs.get("prompt_attention_mask", None)
        pooled_prompt_embeds = kwargs.get("pooled_prompt_embeds", None)
        negative_pooled_prompt_embeds = kwargs.get(
            "negative_pooled_prompt_embeds", None
        )
        ref_latents = kwargs.get("ref_latents", None)
        uncond_ref_latents = kwargs.get("uncond_ref_latents", None)
        timesteps = kwargs.get("timesteps", None)
        num_inference_steps = kwargs.get("num_inference_steps", 50)
        guidance_scale = kwargs.get("guidance_scale", 6.0)
        use_cfg_guidance = kwargs.get("use_cfg_guidance", True)
        dynamic_guidance_start = kwargs.get("dynamic_guidance_start", 3.5)
        dynamic_guidance_end = kwargs.get("dynamic_guidance_end", 6.5)
        motion_exp = kwargs.get("motion_exp", None)
        motion_pose = kwargs.get("motion_pose", None)
        fps_tensor = kwargs.get("fps_tensor", None)
        freqs_cis = kwargs.get("freqs_cis", None)
        transformer_dtype = kwargs.get("transformer_dtype", None)
        hidden_size = kwargs.get("hidden_size", 3072)
        frames_per_batch = kwargs.get("frame_per_batch", 33)
        shift_offset = kwargs.get("shift_offset", 10)
        no_cache_steps = kwargs.get("no_cache_steps", None)
        guidance_rescale = kwargs.get("guidance_rescale", 0.0)
        video_length = kwargs.get("video_length", None)
        shift = 0

        if video_length == frames_per_batch or infer_length == frames_per_batch:
            infer_length = frames_per_batch
            shift_offset = 0
            latents_all = latents_all[:, :, :infer_length]
            audio_prompts_all = audio_prompts_all[:, : infer_length * 4]

        if use_cfg_guidance:
            prompt_embeds_input = torch.cat([negative_prompt_embeds, prompt_embeds])
            prompt_mask_input = torch.cat(
                [negative_prompt_attention_mask, prompt_attention_mask]
            )
            pooled_prompt_embeds_input = torch.cat(
                [negative_pooled_prompt_embeds, pooled_prompt_embeds]
            )
            ref_latents_input = torch.cat([uncond_ref_latents, ref_latents])

        cache_tensor = {}
        with self._progress_bar(
            total=num_inference_steps, desc="Denoising Hunyuan Avatar"
        ) as progress_bar:
            for i, t in enumerate(timesteps):

                pred_latents = torch.zeros_like(latents_all, dtype=transformer_dtype)

                counter = torch.zeros(
                    (latents_all.shape[0], latents_all.shape[1], infer_length, 1, 1),
                    dtype=transformer_dtype,
                ).to(device=latents_all.device)

                for index_start in range(0, infer_length, frames_per_batch):

                    if hasattr(self.scheduler, "_step_index"):
                        self.scheduler._step_index = None

                    index_start = index_start - shift

                    idx_list = [
                        ii % infer_length
                        for ii in range(index_start, index_start + frames_per_batch)
                    ]
                    latents = latents_all[:, :, idx_list].clone()

                    idx_list_audio = [
                        ii % audio_prompts_all.shape[1]
                        for ii in range(
                            index_start * 4, (index_start + frames_per_batch) * 4 - 3
                        )
                    ]

                    # Ensure audio prompt list is not out of bounds
                    if max(idx_list_audio) >= audio_prompts_all.shape[1]:
                        idx_list_audio = [
                            min(i, audio_prompts_all.shape[1] - 1)
                            for i in idx_list_audio
                        ]

                    audio_prompts = audio_prompts_all[:, idx_list_audio].clone()

                    # Classifier-Free Guidance setup
                    if use_cfg_guidance:
                        latent_model_input = torch.cat([latents] * 2)
                    else:
                        latent_model_input = latents
                    latent_model_input = self.scheduler.scale_model_input(
                        latent_model_input, t
                    ).to(transformer_dtype)

                    if use_cfg_guidance:
                        # Dynamic Guidance
                        if i < 10:
                            current_guidance_scale = (1 - i / len(timesteps)) * (
                                guidance_scale - 2
                            ) + 2
                            audio_prompts_input = torch.cat(
                                [uncond_audio_prompts, audio_prompts], dim=0
                            )
                            face_masks_input = torch.cat([face_masks * 0.6] * 2, dim=0)
                        else:
                            current_guidance_scale = (1 - i / len(timesteps)) * (
                                dynamic_guidance_end - dynamic_guidance_start
                            ) + dynamic_guidance_start
                            prompt_embeds_input = torch.cat(
                                [prompt_embeds, prompt_embeds]
                            )
                            prompt_mask_input = torch.cat(
                                [prompt_attention_mask, prompt_attention_mask]
                            )
                            pooled_prompt_embeds_input = torch.cat(
                                [pooled_prompt_embeds, pooled_prompt_embeds]
                            )
                            audio_prompts_input = torch.cat(
                                [uncond_audio_prompts, audio_prompts], dim=0
                            )
                            face_masks_input = torch.cat([face_masks] * 2, dim=0)

                        motion_exp_input = torch.cat([motion_exp] * 2)
                        motion_pose_input = torch.cat([motion_pose] * 2)
                        fps_input = torch.cat([fps_tensor] * 2)
                    else:
                        current_guidance_scale = guidance_scale
                        prompt_embeds_input = prompt_embeds
                        prompt_mask_input = prompt_attention_mask
                        pooled_prompt_embeds_input = pooled_prompt_embeds
                        audio_prompts_input = audio_prompts
                        face_masks_input = face_masks
                        ref_latents_input = ref_latents
                        motion_exp_input = motion_exp
                        motion_pose_input = motion_pose
                        fps_input = fps_tensor

                    latent_input_len = (
                        (latent_model_input.shape[-1] // 2)
                        * (latent_model_input.shape[-2] // 2)
                        * latent_model_input.shape[-3]
                    )
                    latent_ref_len = (
                        (latent_model_input.shape[-1] // 2)
                        * (latent_model_input.shape[-2] // 2)
                        * (latent_model_input.shape[-3] + 1)
                    )

                    timestep = t.repeat(latent_model_input.shape[0])

                    if i in no_cache_steps:
                        use_cache = False

                        with torch.autocast(
                            device_type=self.device.type, dtype=transformer_dtype
                        ):
                            noise_pred = self.transformer(
                                hidden_states=latent_model_input,
                                timestep=timestep,
                                encoder_hidden_states=prompt_embeds_input,
                                encoder_attention_mask=prompt_mask_input,
                                pooled_projections=pooled_prompt_embeds_input,
                                ref_latents=ref_latents_input,
                                encoder_hidden_states_face_mask=face_masks_input,
                                encoder_hidden_states_audio=audio_prompts_input,
                                encoder_hidden_states_motion=motion_exp_input,
                                encoder_hidden_states_pose=motion_pose_input,
                                encoder_hidden_states_fps=fps_input,
                                freqs_cos=freqs_cis[0],
                                freqs_sin=freqs_cis[1],
                                use_cache=False,
                                return_dict=False,
                            )[0]

                        if not cache_tensor:
                            cache_tensor = {
                                "reference_latent": torch.zeros(
                                    [
                                        latent_model_input.shape[0],
                                        latents_all.shape[-3],
                                        (latent_model_input.shape[-1] // 2)
                                        * (latent_model_input.shape[-2] // 2),
                                        hidden_size,
                                    ]
                                )
                                .to(self.transformer.latent_cache.dtype)
                                .to(latent_model_input.device)
                                .clone(),
                                "input_latent": torch.zeros(
                                    [
                                        latent_model_input.shape[0],
                                        latents_all.shape[-3],
                                        (latent_model_input.shape[-1] // 2)
                                        * (latent_model_input.shape[-2] // 2),
                                        hidden_size,
                                    ]
                                )
                                .to(self.transformer.latent_cache.dtype)
                                .to(latent_model_input.device)
                                .clone(),
                                "prompt_embeds": torch.zeros(
                                    [
                                        latent_model_input.shape[0],
                                        latents_all.shape[-3],
                                        prompt_embeds_input.shape[1],
                                        hidden_size,
                                    ]
                                )
                                .to(self.transformer.latent_cache.dtype)
                                .to(latent_model_input.device)
                                .clone(),
                            }

                        cache_tensor["reference_latent"][:, idx_list] = (
                            self.transformer.latent_cache[
                                :, : latent_ref_len - latent_input_len
                            ]
                            .reshape(latent_model_input.shape[0], 1, -1, hidden_size)
                            .repeat(1, len(idx_list), 1, 1)
                            .to(latent_model_input.device)
                        )
                        cache_tensor["input_latent"][:, idx_list] = (
                            self.transformer.latent_cache[
                                :, latent_ref_len - latent_input_len : latent_ref_len
                            ]
                            .reshape(
                                latent_model_input.shape[0],
                                len(idx_list),
                                -1,
                                hidden_size,
                            )
                            .to(latent_model_input.device)
                        )
                        cache_tensor["prompt_embeds"][:, idx_list] = (
                            self.transformer.latent_cache[:, latent_ref_len:]
                            .unsqueeze(1)
                            .repeat(1, len(idx_list), 1, 1)
                            .to(latent_model_input.device)
                        )
                    else:
                        use_cache = True
                        self.transformer.latent_cache[
                            :, : latent_ref_len - latent_input_len
                        ] = cache_tensor["reference_latent"][:, idx_list][:, 0].clone()
                        self.transformer.latent_cache[
                            :, latent_ref_len - latent_input_len : latent_ref_len
                        ] = (
                            cache_tensor["input_latent"][:, idx_list]
                            .reshape(-1, latent_input_len, hidden_size)
                            .clone()
                        )
                        self.transformer.latent_cache[:, latent_ref_len:] = (
                            cache_tensor["prompt_embeds"][:, idx_list][:, 0].clone()
                        )

                        with torch.autocast(
                            device_type=self.device.type, dtype=transformer_dtype
                        ):
                            noise_pred = self.transformer(
                                hidden_states=latent_model_input,
                                timestep=timestep,
                                encoder_hidden_states=prompt_embeds_input,
                                encoder_attention_mask=prompt_mask_input,
                                pooled_projections=pooled_prompt_embeds_input,
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

                    if use_cfg_guidance:
                        # Perform guidance
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + current_guidance_scale * (
                            noise_pred_text - noise_pred_uncond
                        )

                    if use_cfg_guidance and guidance_rescale > 0:
                        noise_pred = self.rescale_noise_cfg(
                            noise_pred,
                            noise_pred_text,
                            guidance_rescale=guidance_rescale,
                        )
                    # Scheduler step
                    latents = self.scheduler.step(
                        noise_pred, t, latents, return_dict=False
                    )[0]

                    latents = latents.to(transformer_dtype)

                    for iii in range(frames_per_batch):
                        p = (index_start + iii) % pred_latents.shape[2]
                        pred_latents[:, :, p] += latents[:, :, iii]
                        counter[:, :, p] += 1

                shift += shift_offset
                shift = shift % frames_per_batch
                pred_latents = pred_latents / counter
                latents_all = pred_latents

                progress_bar.update()

        self.logger.info("Denoising completed.")
        return latents_all
