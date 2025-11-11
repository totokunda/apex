import torch
from src.utils.type import EnumType

class HidreamDenoiseType(EnumType):
    BASE = "base"
    EDIT = "edit"

class HidreamDenoise:
    def __init__(
        self,
        denoise_type: HidreamDenoiseType = HidreamDenoiseType.BASE,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.denoise_type = denoise_type

    def denoise(self, *args, **kwargs):
        if self.denoise_type == HidreamDenoiseType.BASE:
            return self.base_denoise(*args, **kwargs)
        elif self.denoise_type == HidreamDenoiseType.EDIT:
            return self.edit_denoise(*args, **kwargs)
        else:
            raise ValueError(f"Invalid denoise type: {self.denoise_type}")

    def base_denoise(self, *args, **kwargs):
        latents = kwargs.get("latents")
        timesteps = kwargs.get("timesteps")
        num_inference_steps = kwargs.get("num_inference_steps")
        llama_prompt_embeds = kwargs.get("llama_prompt_embeds")
        prompt_embeds = kwargs.get("prompt_embeds")
        pooled_prompt_embeds = kwargs.get("pooled_prompt_embeds")
        num_warmup_steps = kwargs.get("num_warmup_steps")
        use_cfg_guidance = kwargs.get("use_cfg_guidance")
        render_on_step = kwargs.get("render_on_step")
        render_on_step_callback = kwargs.get("render_on_step_callback")
        guidance_scale = kwargs.get("guidance_scale")
        denoise_progress_callback = kwargs.get("denoise_progress_callback", None)

        total_steps = len(timesteps) if timesteps is not None else 0
        if denoise_progress_callback is not None:
            try:
                denoise_progress_callback(0.0, "Starting denoise")
            except Exception:
                pass

        with self._progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):

                # expand the latents if we are doing classifier free guidance
                latent_model_input = (
                    torch.cat([latents] * 2) if use_cfg_guidance else latents
                )
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latent_model_input.shape[0])

                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timesteps=timestep,
                    encoder_hidden_states_t5=prompt_embeds,
                    encoder_hidden_states_llama3=llama_prompt_embeds,
                    pooled_embeds=pooled_prompt_embeds,
                    return_dict=False,
                )[0]
                noise_pred = -noise_pred

                # perform guidance
                if use_cfg_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(
                    noise_pred, t, latents, return_dict=False
                )[0]

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                if render_on_step and render_on_step_callback:
                    self._render_step(latents, render_on_step_callback)

                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()

                if denoise_progress_callback is not None and total_steps > 0:
                    try:
                        denoise_progress_callback(min((i + 1) / total_steps, 1.0), f"Denoising step {i + 1}/{total_steps}")
                    except Exception:
                        pass

        return latents

    def edit_denoise(self, *args, **kwargs):
        latents = kwargs.get("latents")
        timesteps = kwargs.get("timesteps")
        num_inference_steps = kwargs.get("num_inference_steps")
        pooled_prompt_embeds = kwargs.get("pooled_prompt_embeds")
        num_warmup_steps = kwargs.get("num_warmup_steps")
        use_cfg_guidance = kwargs.get("use_cfg_guidance")
        render_on_step = kwargs.get("render_on_step")
        render_on_step_callback = kwargs.get("render_on_step_callback")
        image_latents = kwargs.get("image_latents")
        target_prompt_embeds_t5 = kwargs.get("target_prompt_embeds_t5")
        target_prompt_embeds_llama3 = kwargs.get("target_prompt_embeds_llama3")
        target_pooled_prompt_embeds = kwargs.get("target_pooled_prompt_embeds")
        prompt_embeds_t5 = kwargs.get("prompt_embeds_t5")
        prompt_embeds_llama3 = kwargs.get("prompt_embeds_llama3")
        clip_cfg_norm = kwargs.get("clip_cfg_norm")
        refine_stage = kwargs.get("refine_stage", False)
        refine_strength = kwargs.get("refine_strength", 0.0)
        guidance_scale = kwargs.get("guidance_scale")
        image_guidance_scale = kwargs.get("image_guidance_scale")
        denoise_progress_callback = kwargs.get("denoise_progress_callback", None)

        total_steps = len(timesteps) if timesteps is not None else 0
        if denoise_progress_callback is not None:
            try:
                denoise_progress_callback(0.0, "Starting denoise")
            except Exception:
                pass
        
        if not self.transformer:
            self.load_component_by_type("transformer")
            self.to_device(self.transformer)
        
        self.transformer.max_seq = 8192
            
        with self._progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # === STAGE DETERMINATION ===
                # Check if we need to switch from editing stage to refining stage
                if i == int(num_inference_steps * (1.0 - refine_strength)):
                    refine_stage = True
                
                # === INPUT PREPARATION ===
                if refine_stage:
                    # Refining stage: Use target prompts and simpler input (no image conditioning)
                    latent_model_input_with_condition = torch.cat([latents] * 2) if use_cfg_guidance else latents
                    current_prompt_embeds_t5 = target_prompt_embeds_t5
                    current_prompt_embeds_llama3 = target_prompt_embeds_llama3
                    current_pooled_prompt_embeds = target_pooled_prompt_embeds
                else:
                    # Editing stage: Use original prompts and include image conditioning
                    latent_model_input = torch.cat([latents] * 3) if use_cfg_guidance else latents
                    latent_model_input_with_condition = torch.cat([latent_model_input, image_latents], dim=-1)
                    current_prompt_embeds_t5 = prompt_embeds_t5
                    current_prompt_embeds_llama3 = prompt_embeds_llama3
                    current_pooled_prompt_embeds = pooled_prompt_embeds

                # === TRANSFORMER SELECTION ===
                # Choose which transformer to use for this step
                
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latent_model_input_with_condition.shape[0])

                noise_pred = self.transformer(
                    hidden_states=latent_model_input_with_condition,
                    timesteps=timestep,
                    encoder_hidden_states_t5=current_prompt_embeds_t5,
                    encoder_hidden_states_llama3=current_prompt_embeds_llama3,
                    pooled_embeds=current_pooled_prompt_embeds,
                    return_dict=False,
                )[0]
                # perform guidance
                noise_pred = -1.0 * noise_pred[..., :latents.shape[-1]]
                if use_cfg_guidance:
                    if refine_stage:
                        uncond, full_cond = noise_pred.chunk(2)
                        noise_pred = uncond + guidance_scale * (full_cond - uncond)
                    else:
                        if clip_cfg_norm:
                            uncond, image_cond, full_cond = noise_pred.chunk(3)
                            pred_text_ = image_cond + guidance_scale * (full_cond - image_cond)
                            norm_full_cond = torch.norm(full_cond, dim=1, keepdim=True)
                            norm_pred_text = torch.norm(pred_text_, dim=1, keepdim=True)
                            scale = (norm_full_cond / (norm_pred_text + 1e-8)).clamp(min=0.0, max=1.0)
                            pred_text = pred_text_ * scale
                            noise_pred = uncond + image_guidance_scale * (pred_text - uncond)
                        else:
                            uncond, image_cond, full_cond = noise_pred.chunk(3)
                            noise_pred = uncond + image_guidance_scale * (image_cond - uncond) + guidance_scale * (
                                        full_cond - image_cond)
                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                if render_on_step and render_on_step_callback:
                    self._render_step(latents, render_on_step_callback)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                
                if denoise_progress_callback is not None and total_steps > 0:
                    try:
                        denoise_progress_callback(min((i + 1) / total_steps, 1.0), f"Denoising step {i + 1}/{total_steps}")
                    except Exception:
                        pass
                    
        return latents