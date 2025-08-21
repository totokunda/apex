import torch
from contextlib import nullcontext


class MochiDenoise:
    def denoise(self, *args, **kwargs) -> torch.Tensor:
        timesteps = kwargs.get("timesteps")
        latents = kwargs.get("latents")
        transformer_dtype = kwargs.get("transformer_dtype")
        use_cfg_guidance = kwargs.get("use_cfg_guidance", True)
        render_on_step = kwargs.get("render_on_step", False)
        render_on_step_callback = kwargs.get("render_on_step_callback", None)
        scheduler = kwargs.get("scheduler")
        guidance_scale = kwargs.get("guidance_scale", 5.0)
        num_inference_steps = kwargs.get("num_inference_steps")
        prompt_embeds = kwargs.get("prompt_embeds")
        prompt_attention_mask = kwargs.get("prompt_attention_mask")
        attention_kwargs = kwargs.get("attention_kwargs")
        num_warmup_steps = kwargs.get("num_warmup_steps", 0)

        with self._progress_bar(
            total=num_inference_steps, desc="Denoising Mochi T2V"
        ) as pbar:
            for i, t in enumerate(timesteps):
                if use_cfg_guidance:
                    latent_model_input = torch.cat([latents] * 2).to(transformer_dtype)
                else:
                    latent_model_input = latents.to(transformer_dtype)

                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = (
                    t.expand(latent_model_input.shape[0])
                    .to(self.device)
                    .to(transformer_dtype)
                )

                if hasattr(self.transformer, "cache_context"):
                    cache_context = self.transformer.cache_context("cond_uncond")
                else:
                    cache_context = nullcontext()

                with cache_context:
                    noise_pred = self.transformer(
                        hidden_states=latent_model_input,
                        encoder_hidden_states=prompt_embeds,
                        timestep=timestep,
                        encoder_attention_mask=prompt_attention_mask,
                        attention_kwargs=attention_kwargs,
                        return_dict=False,
                    )[0]

                noise_pred = noise_pred.to(torch.float32)

                if use_cfg_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )

                latents = scheduler.step(
                    noise_pred, t, latents.to(torch.float32), return_dict=False
                )[0].to(transformer_dtype)

                if render_on_step and render_on_step_callback:
                    self._render_step(latents, render_on_step_callback)

                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % scheduler.order == 0
                ):
                    pbar.update(1)

        self.logger.info("Denoising completed.")
        return latents
