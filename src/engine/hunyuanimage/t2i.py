from .base import HunyuanImageBaseEngine
from typing import Union, List, Optional, Dict, Any, TYPE_CHECKING, Callable
import torch
import numpy as np
from diffusers.guiders import AdaptiveProjectedMixGuidance
from src.utils.progress import safe_emit_progress, make_mapped_progress
from src.denoise.hunyuanimage_denoise import HunyuanImageDenoise

if TYPE_CHECKING:
    base_class = HunyuanImageDenoise
else:
    base_class = object

class HunyuanImageT2IEngine(HunyuanImageBaseEngine, base_class):
    def run(self, 
            prompt: Union[List[str], str],
            negative_prompt: Union[List[str], str, None] = None,
            height: int = 768,
            width: int = 1344,
            num_inference_steps: int = 50,
            prompt_embeds: torch.Tensor = None,
            prompt_embeds_mask: torch.Tensor = None,
            prompt_embeds_2: torch.Tensor = None,
            prompt_embeds_mask_2: torch.Tensor = None,
            negative_prompt_embeds: torch.Tensor = None,
            negative_prompt_embeds_mask: torch.Tensor = None,
            negative_prompt_embeds_2: torch.Tensor = None,
            negative_prompt_embeds_mask_2: torch.Tensor = None,
            seed: int = None,
            generator: torch.Generator = None,
            latents: torch.Tensor = None,
            sigmas: List[float] = None,
            timesteps: List[int] = None,
            timesteps_as_indices: bool = True,
            max_sequence_length: int = 1024,
            attention_kwargs: Dict[str, Any] = {},
            num_images: int = 1,
            distilled_guidance_scale: float = 3.25,
            return_latents: bool = False,
            offload: bool = True,
            render_on_step: bool = False,
            render_on_step_callback: Callable = None,
            progress_callback: Callable = None,
            **kwargs,
            ):

        safe_emit_progress(progress_callback, 0.0, "Starting text-to-image pipeline")
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        prompt_embeds, prompt_embeds_mask, prompt_embeds_2, prompt_embeds_mask_2 = self.encode_prompt(
            prompt=prompt,
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            batch_size=batch_size,
            num_images_per_prompt=num_images,
            prompt_embeds_2=prompt_embeds_2,
            prompt_embeds_mask_2=prompt_embeds_mask_2,
            offload=offload,
        )
        safe_emit_progress(progress_callback, 0.15, "Encoded prompt")

        if self.transformer is None:
            self.load_component_by_type("transformer")
        
        self.to_device(self.transformer)
        safe_emit_progress(progress_callback, 0.25, "Transformer ready")
        
        dtype = self.component_dtypes["transformer"]
        
        prompt_embeds = prompt_embeds.to(dtype)
        prompt_embeds_2 = prompt_embeds_2.to(dtype)
        


        # select guider
        if not torch.all(prompt_embeds_2 == 0) and self.helpers["ocr_guider"] is not None:
            # prompt contains ocr and pipeline has a guider for ocr
            guider = self.helpers["ocr_guider"]
        elif self.helpers["guider"] is not None:
            guider = self.helpers["guider"]
        # distilled model does not use guidance method, use default guider with enabled=False
        else:
            guider = AdaptiveProjectedMixGuidance(enabled=False)
       
        if guider._enabled and guider.num_conditions > 1:
            (
                negative_prompt_embeds,
                negative_prompt_embeds_mask,
                negative_prompt_embeds_2,
                negative_prompt_embeds_mask_2,
            ) = self.encode_prompt(
                prompt=negative_prompt,
                prompt_embeds=negative_prompt_embeds,
                prompt_embeds_mask=negative_prompt_embeds_mask,
                batch_size=batch_size,
                num_images_per_prompt=num_images,
                prompt_embeds_2=negative_prompt_embeds_2,
                prompt_embeds_mask_2=negative_prompt_embeds_mask_2,
                offload=offload,
            )

            negative_prompt_embeds = negative_prompt_embeds.to(dtype)
            negative_prompt_embeds_2 = negative_prompt_embeds_2.to(dtype)
            safe_emit_progress(progress_callback, 0.28, "Guidance embeddings prepared")
        else:
            safe_emit_progress(progress_callback, 0.18, "Skipped negative prompt embeds")
    
    
        # 4. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels
        latents = self.get_latents(
            batch_size=batch_size,
            num_channels_latents=num_channels_latents,
            height=height,
            width=width,
            dtype=self.transformer.dtype,
            device=self.device,
            seed=seed,
            generator=generator,
        )
        safe_emit_progress(progress_callback, 0.32, "Initialized latent noise")

        # 5. Prepare timesteps
        if not self.scheduler:
            self.load_component_by_type("scheduler")
        self.to_device(self.scheduler)
        
        safe_emit_progress(progress_callback, 0.36, "Scheduler ready")
        
        sigmas = np.linspace(1.0, 0.0, num_inference_steps + 1)[:-1] if sigmas is None else sigmas
        timesteps, num_inference_steps = self._get_timesteps(self.scheduler, num_inference_steps, sigmas=sigmas)
        safe_emit_progress(progress_callback, 0.40, "Timesteps computed; starting denoise")

        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # handle guidance (for guidance-distilled model)
        if self.transformer.config.guidance_embeds and distilled_guidance_scale is None:
            raise ValueError("`distilled_guidance_scale` is required for guidance-distilled model.")

        if self.transformer.config.guidance_embeds:
            guidance = (
                torch.tensor(
                    [distilled_guidance_scale] * latents.shape[0], dtype=self.transformer.dtype, device=self.device
                )
                * 1000.0
            )

        else:
            guidance = None
        
        denoise_progress_callback = make_mapped_progress(progress_callback, 0.40, 0.92)

        latents = self.denoise(
            timesteps=timesteps,
            latents=latents,
            num_inference_steps=num_inference_steps,
            guidance=guidance,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            negative_prompt_embeds_mask=negative_prompt_embeds_mask,
            prompt_embeds_2=prompt_embeds_2,
            prompt_embeds_mask_2=prompt_embeds_mask_2,
            negative_prompt_embeds_2=negative_prompt_embeds_2,
            negative_prompt_embeds_mask_2=negative_prompt_embeds_mask_2,
            render_on_step=render_on_step,
            render_on_step_callback=render_on_step_callback,
            num_warmup_steps=num_warmup_steps,
            denoise_progress_callback=denoise_progress_callback,
            guider=guider,
            attention_kwargs=attention_kwargs,
        )
        safe_emit_progress(progress_callback, 0.92, "Denoising complete")
        
        if offload:
            self._offload(self.transformer)
            safe_emit_progress(progress_callback, 0.94, "Transformer offloaded")

        if return_latents:
            safe_emit_progress(progress_callback, 1.0, "Returning latents")
            return latents
        else:
            image = self.vae_decode(latents, offload=offload)
            image = self._tensor_to_frame(image)
            safe_emit_progress(progress_callback, 1.0, "Completed text-to-image pipeline")
            return image