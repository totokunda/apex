import torch
from src.utils.type import EnumType


class QwenImageDenoiseType(EnumType):
    BASE = "base"
    CONTROLNET = "controlnet"

class QwenImageDenoise:
    def __init__(
        self,
        denoise_type: QwenImageDenoiseType = QwenImageDenoiseType.BASE,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.denoise_type = denoise_type

    def denoise(self, *args, **kwargs):
        if self.denoise_type == QwenImageDenoiseType.BASE:
            return self.base_denoise(*args, **kwargs)
        elif self.denoise_type == QwenImageDenoiseType.CONTROLNET:
            return self.controlnet_denoise(*args, **kwargs)
        else:
            raise ValueError(f"Invalid denoise type: {self.denoise_type}")

    def base_denoise(self, *args, **kwargs):
        latents = kwargs.get("latents")
        timesteps = kwargs.get("timesteps")
        num_inference_steps = kwargs.get("num_inference_steps")
        guidance = kwargs.get("guidance")
        prompt_embeds = kwargs.get("prompt_embeds")
        image_latents = kwargs.get("image_latents", None)
        prompt_embeds_mask = kwargs.get("prompt_embeds_mask")
        img_shapes = kwargs.get("img_shapes")
        use_cfg_guidance = kwargs.get("use_cfg_guidance")
        negative_prompt_embeds = kwargs.get("negative_prompt_embeds")
        negative_prompt_embeds_mask = kwargs.get("negative_prompt_embeds_mask")
        txt_seq_lens = kwargs.get("txt_seq_lens")
        negative_txt_seq_lens = kwargs.get("negative_txt_seq_lens")
        attention_kwargs = kwargs.get("attention_kwargs")
        render_on_step = kwargs.get("render_on_step")
        render_on_step_callback = kwargs.get("render_on_step_callback")
        num_warmup_steps = kwargs.get("num_warmup_steps")
        true_cfg_scale = kwargs.get("true_cfg_scale")
        mask_image_latents = kwargs.get("mask_image_latents", None)
        mask = kwargs.get("mask", None)
        noise = kwargs.get("noise", None)

        with self._progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latents.shape[0]).to(latents.dtype)
                
                if image_latents is not None:
                    latent_model_input = torch.cat([latents, image_latents], dim=1)
                else:
                    latent_model_input = latents
                    
                with self.transformer.cache_context("cond"):
                    noise_pred = self.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep / 1000,
                        guidance=guidance,
                        encoder_hidden_states_mask=prompt_embeds_mask,
                        encoder_hidden_states=prompt_embeds,
                        img_shapes=img_shapes,
                        txt_seq_lens=txt_seq_lens,
                        attention_kwargs=attention_kwargs,
                        return_dict=False,
                    )[0]
                    if image_latents is not None:
                        noise_pred = noise_pred[:, : latents.size(1)]

                if use_cfg_guidance:
                    with self.transformer.cache_context("uncond"):
                        neg_noise_pred = self.transformer(
                            hidden_states=latent_model_input,
                            timestep=timestep / 1000,
                            guidance=guidance,
                            encoder_hidden_states_mask=negative_prompt_embeds_mask,
                            encoder_hidden_states=negative_prompt_embeds,
                            img_shapes=img_shapes,
                            txt_seq_lens=negative_txt_seq_lens,
                            attention_kwargs=attention_kwargs,
                            return_dict=False,
                        )[0]
                        if image_latents is not None:
                            neg_noise_pred = neg_noise_pred[:, : latents.size(1)]
                    comb_pred = neg_noise_pred + true_cfg_scale * (
                        noise_pred - neg_noise_pred
                    )

                    cond_norm = torch.norm(noise_pred, dim=-1, keepdim=True)
                    noise_norm = torch.norm(comb_pred, dim=-1, keepdim=True)
                    noise_pred = comb_pred * (cond_norm / noise_norm)

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(
                    noise_pred, t, latents, return_dict=False
                )[0]
                
                if mask_image_latents is not None and mask is not None:
                    init_latents_proper = mask_image_latents
                    init_mask = mask
                    if i < len(timesteps) - 1:
                        noise_timestep = timesteps[i + 1]
                        init_latents_proper = self.scheduler.scale_noise(
                            init_latents_proper, torch.tensor([noise_timestep]), noise
                        )

                    latents = (1 - init_mask) * init_latents_proper + init_mask * latents

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        latents = latents.to(latents_dtype)

                if render_on_step and render_on_step_callback:
                    self._render_step(latents, render_on_step_callback)

                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()

        return latents
    
    
    def controlnet_denoise(self, *args, **kwargs):
        latents = kwargs.get("latents")
        timesteps = kwargs.get("timesteps")
        num_inference_steps = kwargs.get("num_inference_steps")
        guidance = kwargs.get("guidance")
        prompt_embeds = kwargs.get("prompt_embeds")
        image_latents = kwargs.get("image_latents")
        prompt_embeds_mask = kwargs.get("prompt_embeds_mask")
        img_shapes = kwargs.get("img_shapes")
        use_cfg_guidance = kwargs.get("use_cfg_guidance")
        negative_prompt_embeds = kwargs.get("negative_prompt_embeds")
        negative_prompt_embeds_mask = kwargs.get("negative_prompt_embeds_mask")
        txt_seq_lens = kwargs.get("txt_seq_lens")
        negative_txt_seq_lens = kwargs.get("negative_txt_seq_lens")
        attention_kwargs = kwargs.get("attention_kwargs")
        render_on_step = kwargs.get("render_on_step")
        render_on_step_callback = kwargs.get("render_on_step_callback")
        num_warmup_steps = kwargs.get("num_warmup_steps")
        true_cfg_scale = kwargs.get("true_cfg_scale")
        controlnet_keep = kwargs.get("controlnet_keep")
        controlnet_conditioning_scale = kwargs.get("controlnet_conditioning_scale")
        control_image = kwargs.get("control_image")
        
        with self._progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latents.shape[0]).to(latents.dtype)
                
                if isinstance(controlnet_keep[i], list):
                    cond_scale = [c * s for c, s in zip(controlnet_conditioning_scale, controlnet_keep[i])]
                else:
                    controlnet_cond_scale = controlnet_conditioning_scale
                    if isinstance(controlnet_cond_scale, list):
                        controlnet_cond_scale = controlnet_cond_scale[0]
                    cond_scale = controlnet_cond_scale * controlnet_keep[i]
                    
                controlnet_block_samples = self.controlnet(
                    hidden_states=latents,
                    controlnet_cond=control_image,
                    conditioning_scale=cond_scale,
                    timestep=timestep / 1000,
                    encoder_hidden_states=prompt_embeds,
                    encoder_hidden_states_mask=prompt_embeds_mask,
                    img_shapes=img_shapes,
                    txt_seq_lens=txt_seq_lens,
                    return_dict=False,
                )
                    
                with self.transformer.cache_context("cond"):
                    noise_pred = self.transformer(
                        hidden_states=latents,
                        timestep=timestep / 1000,
                        guidance=guidance,
                        encoder_hidden_states_mask=prompt_embeds_mask,
                        encoder_hidden_states=prompt_embeds,
                        img_shapes=img_shapes,
                        txt_seq_lens=txt_seq_lens,
                        controlnet_block_samples=controlnet_block_samples,
                        attention_kwargs=attention_kwargs,
                        return_dict=False,
                    )[0]
 
                if use_cfg_guidance:
                    with self.transformer.cache_context("uncond"):
                        neg_noise_pred = self.transformer(
                            hidden_states=latents,
                            timestep=timestep / 1000,
                            guidance=guidance,
                            encoder_hidden_states_mask=negative_prompt_embeds_mask,
                            encoder_hidden_states=negative_prompt_embeds,
                            img_shapes=img_shapes,
                            txt_seq_lens=negative_txt_seq_lens,
                            controlnet_block_samples=controlnet_block_samples,
                            attention_kwargs=attention_kwargs,
                            return_dict=False,
                        )[0]

                    comb_pred = neg_noise_pred + true_cfg_scale * (
                        noise_pred - neg_noise_pred
                    )

                    cond_norm = torch.norm(noise_pred, dim=-1, keepdim=True)
                    noise_norm = torch.norm(comb_pred, dim=-1, keepdim=True)
                    noise_pred = comb_pred * (cond_norm / noise_norm)

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(
                    noise_pred, t, latents, return_dict=False
                )[0]

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        latents = latents.to(latents_dtype)

                if render_on_step and render_on_step_callback:
                    self._render_step(latents, render_on_step_callback)

                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()

        return latents
