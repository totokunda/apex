import torch
from src.utils.type import EnumType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.engine.base_engine import BaseEngine  # noqa: F401
    BaseClass = BaseEngine  # type: ignore
else:
    BaseClass = object

class HunyuanImageDenoiseType(EnumType):
    BASE = "base"

class HunyuanImageDenoise(BaseClass):
    def __init__(
        self,
        denoise_type: HunyuanImageDenoiseType = HunyuanImageDenoiseType.BASE,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.denoise_type = denoise_type

    def denoise(self, *args, **kwargs):
        if self.denoise_type == HunyuanImageDenoiseType.BASE:
            return self.base_denoise(*args, **kwargs)
        
    def base_denoise(self, *args, **kwargs):
        timesteps = kwargs.get("timesteps")
        latents = kwargs.get("latents")
        num_inference_steps = kwargs.get("num_inference_steps")
        guidance = kwargs.get("guidance")
        prompt_embeds = kwargs.get("prompt_embeds")
        negative_prompt_embeds = kwargs.get("negative_prompt_embeds")
        prompt_embeds_mask = kwargs.get("prompt_embeds_mask")
        negative_prompt_embeds_mask = kwargs.get("negative_prompt_embeds_mask")
        prompt_embeds_2 = kwargs.get("prompt_embeds_2")
        prompt_embeds_mask_2 = kwargs.get("prompt_embeds_mask_2")
        negative_prompt_embeds_2 = kwargs.get("negative_prompt_embeds_2")
        negative_prompt_embeds_mask_2 = kwargs.get("negative_prompt_embeds_mask_2")
        render_on_step = kwargs.get("render_on_step")
        render_on_step_callback = kwargs.get("render_on_step_callback")
        num_warmup_steps = kwargs.get("num_warmup_steps")
        denoise_progress_callback = kwargs.get("denoise_progress_callback", None)
        guider = kwargs.get("guider")
        attention_kwargs = kwargs.get("attention_kwargs")
        
        self.scheduler.set_begin_index(0)
        total_steps = len(timesteps) if timesteps is not None else 0
        if denoise_progress_callback is not None:
            try:
                denoise_progress_callback(0.0, "Starting denoise")
            except Exception:
                pass
        with self._progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latents.shape[0]).to(latents.dtype)

                if self.transformer.config.use_meanflow:
                    if i == len(timesteps) - 1:
                        timestep_r = torch.tensor([0.0], device=self.device)
                    else:
                        timestep_r = timesteps[i + 1]
                    timestep_r = timestep_r.expand(latents.shape[0]).to(latents.dtype)
                else:
                    timestep_r = None

                # Step 1: Collect model inputs needed for the guidance method
                # conditional inputs should always be first element in the tuple
                guider_inputs = {
                    "encoder_hidden_states": (prompt_embeds, negative_prompt_embeds),
                    "encoder_attention_mask": (prompt_embeds_mask, negative_prompt_embeds_mask),
                    "encoder_hidden_states_2": (prompt_embeds_2, negative_prompt_embeds_2),
                    "encoder_attention_mask_2": (prompt_embeds_mask_2, negative_prompt_embeds_mask_2),
                }

                # Step 2: Update guider's internal state for this denoising step
                guider.set_state(step=i, num_inference_steps=num_inference_steps, timestep=t)

                # Step 3: Prepare batched model inputs based on the guidance method
                # The guider splits model inputs into separate batches for conditional/unconditional predictions.
                # For CFG with guider_inputs = {"encoder_hidden_states": (prompt_embeds, negative_prompt_embeds)}:
                # you will get a guider_state with two batches:
                #   guider_state = [
                #       {"encoder_hidden_states": prompt_embeds, "__guidance_identifier__": "pred_cond"},      # conditional batch
                #       {"encoder_hidden_states": negative_prompt_embeds, "__guidance_identifier__": "pred_uncond"},  # unconditional batch
                #   ]
                # Other guidance methods may return 1 batch (no guidance) or 3+ batches (e.g., PAG, APG).
                guider_state = guider.prepare_inputs(guider_inputs)
                # Step 4: Run the denoiser for each batch
                # Each batch in guider_state represents a different conditioning (conditional, unconditional, etc.).
                # We run the model once per batch and store the noise prediction in guider_state_batch.noise_pred.
                for guider_state_batch in guider_state:
                    guider.prepare_models(self.transformer)

                    # Extract conditioning kwargs for this batch (e.g., encoder_hidden_states)
                    cond_kwargs = {
                        input_name: getattr(guider_state_batch, input_name) for input_name in guider_inputs.keys()
                    }

                    # e.g. "pred_cond"/"pred_uncond"
                    context_name = getattr(guider_state_batch, guider._identifier_key)
                    with self.transformer.cache_context(context_name):
                        # Run denoiser and store noise prediction in this batch
                        guider_state_batch.noise_pred = self.transformer(
                            hidden_states=latents,
                            timestep=timestep,
                            timestep_r=timestep_r,
                            guidance=guidance,
                            attention_kwargs=attention_kwargs,
                            return_dict=False,
                            **cond_kwargs,
                        )[0]

                    # Cleanup model (e.g., remove hooks)
                    guider.cleanup_models(self.transformer)

                # Step 5: Combine predictions using the guidance method
                # The guider takes all noise predictions from guider_state and combines them according to the guidance algorithm.
                # Continuing the CFG example, the guider receives:
                #   guider_state = [
                #       {"encoder_hidden_states": prompt_embeds, "noise_pred": noise_pred_cond, "__guidance_identifier__": "pred_cond"},      # batch 0
                #       {"encoder_hidden_states": negative_prompt_embeds, "noise_pred": noise_pred_uncond, "__guidance_identifier__": "pred_uncond"},  # batch 1
                #   ]
                # And extracts predictions using the __guidance_identifier__:
                #   pred_cond = guider_state[0]["noise_pred"]      # extracts noise_pred_cond
                #   pred_uncond = guider_state[1]["noise_pred"]    # extracts noise_pred_uncond
                # Then applies CFG formula:
                #   noise_pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)
                # Returns GuiderOutput(pred=noise_pred, pred_cond=pred_cond, pred_uncond=pred_uncond)
                noise_pred = guider(guider_state)[0]

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
        denoise_progress_callback = kwargs.get("denoise_progress_callback", None)

        total_steps = len(timesteps) if timesteps is not None else 0
        if denoise_progress_callback is not None:
            try:
                denoise_progress_callback(0.0, "Starting denoise")
            except Exception:
                pass
        
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

                if denoise_progress_callback is not None and total_steps > 0:
                    try:
                        denoise_progress_callback(min((i + 1) / total_steps, 1.0), f"Denoising step {i + 1}/{total_steps}")
                    except Exception:
                        pass

        return latents
