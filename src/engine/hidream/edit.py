import torch
from typing import Dict, Any, Callable, List
import numpy as np
from diffusers.schedulers import UniPCMultistepScheduler
from .base import HidreamBaseEngine
from src.utils.cache import empty_cache
import math

class HidreamEditEngine(HidreamBaseEngine):
    """Hidream Edit Engine Implementation"""

    def run(
        self,
        image: torch.Tensor | str | np.ndarray,
        prompt: List[str] | str,
        prompt_2: List[str] | str = None,
        prompt_3: List[str] | str = None,
        prompt_4: List[str] | str = None,
        negative_prompt: List[str] | str = None,
        negative_prompt_2: List[str] | str = None,
        negative_prompt_3: List[str] | str = None,
        negative_prompt_4: List[str] | str = None,
        num_inference_steps: int = 30,
        num_images: int = 1,
        seed: int | None = None,
        guidance_scale: float = 5.0,
        image_guidance_scale: float = 2.0,
        return_latents: bool = False,
        text_encoder_kwargs: Dict[str, Any] = {},
        text_encoder_2_kwargs: Dict[str, Any] = {},
        text_encoder_3_kwargs: Dict[str, Any] = {},
        joint_attention_kwargs: Dict[str, Any] = {},
        render_on_step_callback: Callable = None,
        offload: bool = True,
        render_on_step: bool = False,
        generator: torch.Generator | None = None,
        sigmas: List[float] | None = None,
        timesteps: List[int] | None = None,
        clip_cfg_norm: bool = True,
        refine_strength: float = 0.0,
        resize_to: int = 1024,
        **kwargs,
    ):

        if not hasattr(self, "text_encoder") or not self.text_encoder:
            self.load_component_by_name("text_encoder")
        self.to_device(self.text_encoder)

        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        use_cfg_guidance = guidance_scale > 1.0
        batch_size = num_images * len(prompt) if isinstance(prompt, list) else num_images


        (
            prompt_embeds_t5,
            negative_prompt_embeds_t5,
            prompt_embeds_llama3,
            negative_prompt_embeds_llama3,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_3=prompt_3,
            prompt_4=prompt_4,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            negative_prompt_3=negative_prompt_3,
            negative_prompt_4=negative_prompt_4,
            text_encoder_kwargs=text_encoder_kwargs,
            text_encoder_2_kwargs=text_encoder_2_kwargs,
            text_encoder_3_kwargs=text_encoder_3_kwargs,
            num_images=num_images,
            use_cfg_guidance=use_cfg_guidance,
            offload=offload,
        )
        
        if "Target Image Description:" in prompt:
            target_prompt = prompt.split("Target Image Description:")[1].strip()
            (
            target_prompt_embeds_t5,
            target_negative_prompt_embeds_t5,
            target_prompt_embeds_llama3,
            target_negative_prompt_embeds_llama3,
            target_pooled_prompt_embeds,
            target_negative_pooled_prompt_embeds,
            ) = self.encode_prompt(
                prompt=target_prompt,
                prompt_2=None,
                prompt_3=None,
                prompt_4=None,
                negative_prompt=negative_prompt,
                negative_prompt_2=None,
                negative_prompt_3=None,
                negative_prompt_4=None,
                use_cfg_guidance=use_cfg_guidance,
                num_images=num_images,
                text_encoder_kwargs=text_encoder_kwargs,
                text_encoder_2_kwargs=text_encoder_2_kwargs,
                text_encoder_3_kwargs=text_encoder_3_kwargs,
                offload=offload,
            )
        else:
            target_prompt_embeds_t5 = prompt_embeds_t5
            target_negative_prompt_embeds_t5 = negative_prompt_embeds_t5
            target_prompt_embeds_llama3 = prompt_embeds_llama3
            target_negative_prompt_embeds_llama3 = negative_prompt_embeds_llama3
            target_pooled_prompt_embeds = pooled_prompt_embeds
            target_negative_pooled_prompt_embeds = negative_pooled_prompt_embeds 

        transformer_dtype = self.component_dtypes.get("transformer", None)

        if use_cfg_guidance:
            if clip_cfg_norm:
                prompt_embeds_t5 = torch.cat([prompt_embeds_t5, negative_prompt_embeds_t5, prompt_embeds_t5], dim=0)
                prompt_embeds_llama3 = torch.cat([prompt_embeds_llama3, negative_prompt_embeds_llama3, prompt_embeds_llama3], dim=1)
                pooled_prompt_embeds = torch.cat([pooled_prompt_embeds, negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
            else:
                prompt_embeds_t5 = torch.cat([negative_prompt_embeds_t5, negative_prompt_embeds_t5, prompt_embeds_t5], dim=0)
                prompt_embeds_llama3 = torch.cat([negative_prompt_embeds_llama3, negative_prompt_embeds_llama3, prompt_embeds_llama3], dim=1)
                pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
            
            target_prompt_embeds_t5 = torch.cat([target_negative_prompt_embeds_t5, target_prompt_embeds_t5], dim=0)
            target_prompt_embeds_llama3 = torch.cat([target_negative_prompt_embeds_llama3, target_prompt_embeds_llama3], dim=1)
            target_pooled_prompt_embeds = torch.cat([target_negative_pooled_prompt_embeds, target_pooled_prompt_embeds], dim=0)
        
        image = self._load_image(image)
        image = self.resize_image(image, image_size=resize_to)
        image = self.image_processor.preprocess(image)
        
        image_latents = self.vae_encode(image, offload=offload)
        latent_height, latent_width = image_latents.shape[2:]
        height = latent_height * self.vae_scale_factor
        width = latent_width * self.vae_scale_factor
        

        if image_latents.shape[0] != batch_size:
            additional_image_per_prompt = batch_size // image_latents.shape[0]
            image_latents = torch.cat([image_latents] * additional_image_per_prompt)
        else:
            image_latents = torch.cat([image_latents])
        
        if use_cfg_guidance:
            uncond_image_latents = torch.zeros_like(image_latents)
            image_latents = torch.cat([uncond_image_latents, image_latents, image_latents], dim=0)
        
        latents = self._get_latents(
            batch_size=batch_size,
            num_channels_latents=self.num_channels_latents,
            height=height,
            width=width,
            dtype=transformer_dtype,
            device=self.device,
            generator=generator,
        )
        
        if not self.scheduler:
            self.load_component_by_type("scheduler")
        self.to_device(self.scheduler)

        max_seq = 8192
        if not isinstance(self.scheduler, UniPCMultistepScheduler):
            sigmas = (
                np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
                if sigmas is None
                else sigmas
            )
            mu = self.calculate_shift(
                max_seq,
                self.scheduler.config.get("base_image_seq_len", 256),
                self.scheduler.config.get("max_image_seq_len", 4096),
                self.scheduler.config.get("base_shift", 0.5),
                self.scheduler.config.get("max_shift", 1.15),
            )
        else:
            mu = None
            sigmas = None

        timesteps, num_inference_steps = self._get_timesteps(
            self.scheduler,
            num_inference_steps,
            sigmas=sigmas,
            timesteps=timesteps,
            mu=mu,
        )

        num_warmup_steps = max(
            len(timesteps) - num_inference_steps * self.scheduler.order, 0
        )
        

        refine_stage = False

        latents = self.denoise(
            latents=latents,
            timesteps=timesteps,
            num_inference_steps=num_inference_steps,
            prompt_embeds_t5=prompt_embeds_t5,
            prompt_embeds_llama3=prompt_embeds_llama3,
            image_latents=image_latents,
            refine_stage=refine_stage,
            target_prompt_embeds_t5=target_prompt_embeds_t5,
            target_prompt_embeds_llama3=target_prompt_embeds_llama3,
            target_pooled_prompt_embeds=target_pooled_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            use_cfg_guidance=use_cfg_guidance,
            guidance_scale=guidance_scale,
            joint_attention_kwargs=joint_attention_kwargs,            
            render_on_step=render_on_step,
            render_on_step_callback=render_on_step_callback,
            num_warmup_steps=num_warmup_steps,
            clip_cfg_norm=clip_cfg_norm,
            refine_strength=refine_strength,
            image_guidance_scale=image_guidance_scale,
        )
        
        if offload:
            self._offload(self.transformer)

        if return_latents:
            return latents

        image = self.vae_decode(latents, offload=offload)
        image = self._tensor_to_frame(image)
        return [image]
