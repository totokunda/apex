import torch
from typing import Dict, Any, Callable, List
import numpy as np
from diffusers.schedulers import UniPCMultistepScheduler
from .base import HidreamBaseEngine
import math


class HidreamT2IEngine(HidreamBaseEngine):
    """Hidream Text-to-Image Engine Implementation"""

    def run(
        self,
        prompt: List[str] | str,
        prompt_2: List[str] | str = None,
        prompt_3: List[str] | str = None,
        prompt_4: List[str] | str = None,
        negative_prompt: List[str] | str = None,
        negative_prompt_2: List[str] | str = None,
        negative_prompt_3: List[str] | str = None,
        negative_prompt_4: List[str] | str = None,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 30,
        num_images: int = 1,
        seed: int | None = None,
        guidance_scale: float = 5.0,
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
        **kwargs,
    ):

        if not hasattr(self, "text_encoder") or not self.text_encoder:
            self.load_component_by_name("text_encoder")

        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        self.to_device(self.text_encoder)
        
        batch_size = num_images * len(prompt) if isinstance(prompt, list) else num_images   

        use_cfg_guidance = guidance_scale > 1.0

        division = self.vae_scale_factor * 2
        S_max = (self.default_sample_size * self.vae_scale_factor) ** 2
        scale = S_max / (width * height)
        scale = math.sqrt(scale)
        width, height = int(width * scale // division * division), int(
            height * scale // division * division
        )

        (
            prompt_embeds,
            negative_prompt_embeds,
            llama_prompt_embeds,
            llama_negative_prompt_embeds,
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

        transformer_dtype = self.component_dtypes.get("transformer", None)
        

        if use_cfg_guidance:
            llama_prompt_embeds = torch.cat([llama_negative_prompt_embeds.to(self.device), llama_prompt_embeds.to(self.device)], dim=1).to(transformer_dtype)
            prompt_embeds = torch.cat([negative_prompt_embeds.to(self.device), prompt_embeds.to(self.device)], dim=0).to(transformer_dtype)
            pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds.to(self.device), pooled_prompt_embeds.to(self.device)], dim=0).to(transformer_dtype)
    
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

        if not hasattr(self, "transformer") or not self.transformer:
            self.load_component_by_type("transformer")

        self.to_device(self.transformer)

        if not isinstance(self.scheduler, UniPCMultistepScheduler):
            sigmas = (
                np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
                if sigmas is None
                else sigmas
            )
            mu = self.calculate_shift(
                self.transformer.max_seq,
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

        latents = self.denoise(
            latents=latents,
            timesteps=timesteps,
            num_inference_steps=num_inference_steps,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            use_cfg_guidance=use_cfg_guidance,
            joint_attention_kwargs=joint_attention_kwargs,
            llama_prompt_embeds=llama_prompt_embeds,
            render_on_step=render_on_step,
            render_on_step_callback=render_on_step_callback,
            num_warmup_steps=num_warmup_steps,
            guidance_scale=guidance_scale,
        )

        if return_latents:
            return latents

        image = self.vae_decode(latents, offload=offload)
        image = self._tensor_to_frame(image)
        return [image]
