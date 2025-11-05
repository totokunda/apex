import torch
from src.utils.type import EnumType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.engine.base_engine import BaseEngine  # noqa: F401
    BaseClass = BaseEngine  # type: ignore
else:
    BaseClass = object

class ChromaDenoiseType(EnumType):
    BASE = "base"


class ChromaDenoise(BaseClass):
    def __init__(
        self,
        denoise_type: ChromaDenoiseType = ChromaDenoiseType.BASE,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.denoise_type = denoise_type

    def denoise(self, *args, **kwargs):
        if self.denoise_type == ChromaDenoiseType.BASE:
            return self.base_denoise(*args, **kwargs)
        else:
            raise ValueError(f"Invalid denoise type: {self.denoise_type}")

    def base_denoise(self, *args, **kwargs):
        latents = kwargs.get("latents")
        timesteps = kwargs.get("timesteps")
        num_inference_steps = kwargs.get("num_inference_steps")
        guidance_scale = kwargs.get("guidance_scale")
        prompt_embeds = kwargs.get("prompt_embeds")
        negative_prompt_embeds = kwargs.get("negative_prompt_embeds")
        attention_mask = kwargs.get("attention_mask")
        negative_attention_mask = kwargs.get("negative_attention_mask")
        render_on_step = kwargs.get("render_on_step")
        render_on_step_callback = kwargs.get("render_on_step_callback")
        num_warmup_steps = kwargs.get("num_warmup_steps")
        image_embeds = kwargs.get("image_embeds")
        negative_image_embeds = kwargs.get("negative_image_embeds")
        text_ids = kwargs.get("text_ids")
        negative_text_ids = kwargs.get("negative_text_ids")
        latent_image_ids = kwargs.get("latent_image_ids")
        joint_attention_kwargs = kwargs.get("joint_attention_kwargs")
        use_cfg_guidance = kwargs.get("use_cfg_guidance")
        denoise_progress_callback = kwargs.get("denoise_progress_callback", None)

        total_steps = len(timesteps) if timesteps is not None else 0
        if denoise_progress_callback is not None:
            try:
                denoise_progress_callback(0.0, "Starting denoise")
            except Exception:
                pass
        
        with self._progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if image_embeds is not None:
                    joint_attention_kwargs["ip_adapter_image_embeds"] = image_embeds

                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latents.shape[0]).to(latents.dtype)

                noise_pred = self.transformer(
                    hidden_states=latents,
                    timestep=timestep / 1000,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    attention_mask=attention_mask,
                    joint_attention_kwargs=joint_attention_kwargs,
                    return_dict=False,
                )[0]

                if use_cfg_guidance:    
                    if negative_image_embeds is not None:
                        joint_attention_kwargs["ip_adapter_image_embeds"] = negative_image_embeds
                    neg_noise_pred = self.transformer(
                        hidden_states=latents,
                        timestep=timestep / 1000,
                        encoder_hidden_states=negative_prompt_embeds,
                        txt_ids=negative_text_ids,
                        img_ids=latent_image_ids,
                        attention_mask=negative_attention_mask,
                        joint_attention_kwargs=joint_attention_kwargs,
                        return_dict=False,
                    )[0]
                    noise_pred = neg_noise_pred + guidance_scale * (noise_pred - neg_noise_pred)

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
