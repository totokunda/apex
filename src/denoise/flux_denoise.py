import torch
from src.utils.type import EnumType
from src.utils.progress import safe_emit_progress


class FluxDenoiseType(EnumType):
    BASE = "base"


class FluxDenoise:
    def __init__(
        self, denoise_type: FluxDenoiseType = FluxDenoiseType.BASE, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.denoise_type = denoise_type

    def denoise(self, *args, **kwargs):
        if self.denoise_type == FluxDenoiseType.BASE:
            return self.base_denoise(*args, **kwargs)
        else:
            raise ValueError(f"Invalid denoise type: {self.denoise_type}")

    def base_denoise(self, *args, **kwargs):
        latents = kwargs.get("latents")
        timesteps = kwargs.get("timesteps")
        num_inference_steps = kwargs.get("num_inference_steps")
        guidance = kwargs.get("guidance")
        prompt_embeds = kwargs.get("prompt_embeds")
        pooled_prompt_embeds = kwargs.get("pooled_prompt_embeds")
        negative_prompt_embeds = kwargs.get("negative_prompt_embeds")
        negative_pooled_prompt_embeds = kwargs.get("negative_pooled_prompt_embeds")
        true_cfg_scale = kwargs.get("true_cfg_scale")
        latent_ids = kwargs.get("latent_ids")
        text_ids = kwargs.get("text_ids")
        negative_text_ids = kwargs.get("negative_text_ids")
        image_embeds = kwargs.get("image_embeds", None)
        negative_image_embeds = kwargs.get("negative_image_embeds", None)
        num_warmup_steps = kwargs.get("num_warmup_steps")
        use_cfg_guidance = kwargs.get("use_cfg_guidance")
        joint_attention_kwargs = kwargs.get("joint_attention_kwargs")
        render_on_step = kwargs.get("render_on_step")
        render_on_step_callback = kwargs.get("render_on_step_callback")
        image_latents = kwargs.get("image_latents")
        concat_latents = kwargs.get("concat_latents")
        denoise_progress_callback = kwargs.get("denoise_progress_callback")

        safe_emit_progress(denoise_progress_callback, 0.0, "Starting denoise")
        with self._progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):

                if image_embeds is not None:
                    joint_attention_kwargs["ip_adapter_image_embeds"] = image_embeds
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latents.shape[0]).to(latents.dtype)
                
                if image_latents is not None:
                    latent_model_input = torch.cat([latents, image_latents], dim=1)
                elif concat_latents is not None:
                    latent_model_input = torch.cat([latents, concat_latents], dim=2)
                else:
                    latent_model_input = latents

                with self.transformer.cache_context("cond"):
                    noise_pred = self.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep / 1000,
                        guidance=guidance,
                        pooled_projections=pooled_prompt_embeds,
                        encoder_hidden_states=prompt_embeds,
                        txt_ids=text_ids,
                        img_ids=latent_ids,
                        joint_attention_kwargs=joint_attention_kwargs,
                        return_dict=False,
                    )[0]
                    
                    if image_latents is not None:
                        noise_pred = noise_pred[:, : latents.size(1)]

                if use_cfg_guidance:
                    if negative_image_embeds is not None:
                        joint_attention_kwargs["ip_adapter_image_embeds"] = (
                            negative_image_embeds
                        )

                    with self.transformer.cache_context("uncond"):
                        neg_noise_pred = self.transformer(
                            hidden_states=latent_model_input,
                            timestep=timestep / 1000,
                            guidance=guidance,
                            pooled_projections=negative_pooled_prompt_embeds,
                            encoder_hidden_states=negative_prompt_embeds,
                            txt_ids=negative_text_ids,
                            img_ids=latent_ids,
                            joint_attention_kwargs=joint_attention_kwargs,
                            return_dict=False,
                        )[0]
                        if image_latents is not None:
                            neg_noise_pred = neg_noise_pred[:, : latents.size(1)]
                        
                    noise_pred = neg_noise_pred + true_cfg_scale * (
                        noise_pred - neg_noise_pred
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
                
                # external progress callback
                safe_emit_progress(
                    denoise_progress_callback,
                    float(i + 1) / float(len(timesteps)) if len(timesteps) > 0 else 1.0,
                    f"Denoise {i + 1}/{len(timesteps)}",
                )

        safe_emit_progress(denoise_progress_callback, 1.0, "Denoise finished")
        return latents
