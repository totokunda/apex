import torch
from src.utils.type import EnumType


class HidreamDenoiseType(EnumType):
    BASE = "base"


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

        return latents
