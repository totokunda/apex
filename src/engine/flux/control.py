import torch
from typing import Dict, Any, Callable, List, Union, Optional
from PIL import Image
import numpy as np
from .base import FluxBaseEngine

class FluxControlEngine(FluxBaseEngine):
    """Flux Control Engine Implementation"""

    def run(
        self,
        control_image: Image.Image | str | np.ndarray | torch.Tensor,
        prompt: List[str] | str,
        prompt_2: List[str] | str = None,
        negative_prompt: List[str] | str = None,
        negative_prompt_2: List[str] | str = None,
        height: int | None = None,
        width: int | None = None,
        aspect_ratio: str = "1:1",
        resolution: int = 1024,
        num_inference_steps: int = 30,
        num_images: int = 1,
        seed: int | None = None,
        true_cfg_scale: float = 1.0,
        guidance_scale: float = 10.0,
        return_latents: bool = False,
        text_encoder_kwargs: Dict[str, Any] = {},
        text_encoder_2_kwargs: Dict[str, Any] = {},
        joint_attention_kwargs: Dict[str, Any] = {},
        render_on_step_callback: Callable = None,
        offload: bool = True,
        render_on_step: bool = False,
        generator: torch.Generator | None = None,
        sigmas: List[float] | None = None,
        timesteps: List[int] | None = None,
        **kwargs,
    ):
        
        if height is None and width is None and aspect_ratio is not None:
            height, width = self._aspect_ratio_to_height_width(aspect_ratio, resolution)
        elif height is None and width is None and resolution is not None:
            height, width = self._resolution_to_height_width(resolution)
        elif height is None and width is None:
            height, width = self._image_to_height_width(control_image)

        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
            
        use_cfg_guidance = true_cfg_scale > 1.0 and negative_prompt is not None
            
        pooled_prompt_embeds, negative_pooled_prompt_embeds, prompt_embeds, negative_prompt_embeds, text_ids, negative_text_ids = self.encode_prompt(
            prompt,
            negative_prompt,
            prompt_2,
            negative_prompt_2,
            use_cfg_guidance,
            offload,
            num_images,
            text_encoder_kwargs,
            text_encoder_2_kwargs,
        )
        
        if offload:
            self._offload(self.text_encoder_2)

        transformer_dtype = self.component_dtypes.get("transformer", None)
        batch_size = prompt_embeds.shape[0]
        
        control_image = self._load_image(control_image)
        control_image = self.image_processor.preprocess(control_image, height=height, width=width)
        control_image = control_image.repeat_interleave(batch_size, dim=0)
        
        if use_cfg_guidance:
            control_image = torch.cat([control_image] * 2)
        
        control_image_latents = self.vae_encode(control_image, offload=offload)
        height_control_image, width_control_image = control_image_latents.shape[2:]
        
        control_image_latents = self._pack_latents(
            control_image_latents,
            batch_size,
            self.num_channels_latents,
            height_control_image,
            width_control_image,
        )

        latents, latent_ids = self._get_latents(
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

        sigmas = (
            np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
            if sigmas is None
            else sigmas
        )
        
        image_seq_len = latents.shape[1]
        mu = self.calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )
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

        if not hasattr(self, "transformer") or not self.transformer:
            self.load_component_by_type("transformer")

        self.to_device(self.transformer)

        if self.transformer.config.guidance_embeds:
            guidance = torch.full(
                [1], guidance_scale, device=self.device, dtype=torch.float32
            )
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None

        
        self.scheduler.set_begin_index(0)

        latents = self.denoise(
            latents=latents,
            timesteps=timesteps,
            num_inference_steps=num_inference_steps,
            guidance=guidance,
            prompt_embeds=prompt_embeds,
            concat_latents=control_image_latents,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            use_cfg_guidance=use_cfg_guidance,
            joint_attention_kwargs=joint_attention_kwargs,
            latent_ids=latent_ids,
            text_ids=text_ids,
            negative_text_ids=negative_text_ids,
            render_on_step=render_on_step,
            render_on_step_callback=render_on_step_callback,
            num_warmup_steps=num_warmup_steps,
            true_cfg_scale=true_cfg_scale,
        )

        if return_latents:
            return latents

        latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
        image = self.vae_decode(latents, offload=offload)
        image = self._tensor_to_frame(image)
        return [image]
