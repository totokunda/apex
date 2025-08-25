import torch
from typing import Dict, Any, Callable, List, Optional
from PIL import Image
from .base import ChromaBaseEngine
import numpy as np

class ChromaT2IEngine(ChromaBaseEngine):
    """Chroma Text-to-Image Engine Implementation"""

    def run(
        self,
        prompt: List[str] | str,
        negative_prompt: List[str] | str = None,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 30,
        num_images: int = 1,
        seed: int | None = None,
        guidance_scale: float = 1.0,
        true_cfg_scale: float = 4.0,
        use_cfg_guidance: bool = True,
        return_latents: bool = False,
        text_encoder_kwargs: Dict[str, Any] = {},
        render_on_step_callback: Callable = None,
        offload: bool = True,
        render_on_step: bool = False,
        generator: torch.Generator | None = None,
        timesteps: List[int] | None = None,
        sigmas: List[float] | None = None,
        attention_kwargs: Dict[str, Any] = {},
        ip_adapter_image: Optional[Image.Image | str | np.ndarray] = None,
        ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        negative_ip_adapter_image: Optional[Image.Image | str | np.ndarray] = None,
        negative_ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        **kwargs,
    ):
        if isinstance(prompt, str):
            prompt = [prompt]
        if isinstance(negative_prompt, str):
            negative_prompt = [negative_prompt]

        
        prompt_embeds, prompt_embeds_mask, negative_prompt_embeds, negative_prompt_embeds_mask, text_ids, negative_text_ids = self.encode_prompt(
            prompt,
            negative_prompt,
            num_images=num_images,
            text_encoder_kwargs=text_encoder_kwargs,
            use_cfg_guidance=use_cfg_guidance,
            offload=offload
        )

        transformer_dtype = self.component_dtypes["transformer"]
        prompt_embeds = prompt_embeds.to(self.device, dtype=transformer_dtype)
        prompt_embeds_mask = prompt_embeds_mask.to(self.device)

        if not self.transformer:
            self.load_component_by_type("transformer")

        self.to_device(self.transformer)

        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(
                self.device, dtype=transformer_dtype
            )
            negative_prompt_embeds_mask = negative_prompt_embeds_mask.to(self.device)

        latents, latent_image_ids = self._get_latents(
            batch_size=num_images,
            num_channels_latents=self.num_channels_latents,
            height=height,
            width=width,
            dtype=transformer_dtype,
            device=self.device,
            generator=generator,
            seed=seed,
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
        
        attention_mask = self._prepare_attention_mask(
            batch_size=latents.shape[0],
            sequence_length=image_seq_len,
            dtype=latents.dtype,
            attention_mask=prompt_embeds_mask,
        )
        negative_attention_mask = self._prepare_attention_mask(
            batch_size=latents.shape[0],
            sequence_length=image_seq_len,
            dtype=latents.dtype,
            attention_mask=negative_prompt_embeds_mask,
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
        
        
        if (ip_adapter_image is not None or ip_adapter_image_embeds is not None) and (
            negative_ip_adapter_image is None and negative_ip_adapter_image_embeds is None
        ):
            negative_ip_adapter_image = np.zeros((width, height, 3), dtype=np.uint8)
            negative_ip_adapter_image = [negative_ip_adapter_image] * self.transformer.encoder_hid_proj.num_ip_adapters

        elif (ip_adapter_image is None and ip_adapter_image_embeds is None) and (
            negative_ip_adapter_image is not None or negative_ip_adapter_image_embeds is not None
        ):
            ip_adapter_image = np.zeros((width, height, 3), dtype=np.uint8)
            ip_adapter_image = [ip_adapter_image] * self.transformer.encoder_hid_proj.num_ip_adapters

        image_embeds = None
        negative_image_embeds = None
        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                self.device,
                num_images,
            )
        if negative_ip_adapter_image is not None or negative_ip_adapter_image_embeds is not None:
            negative_image_embeds = self.prepare_ip_adapter_image_embeds(
                negative_ip_adapter_image,
                negative_ip_adapter_image_embeds,
                self.device,
                num_images,
            )
            
        

        # 6. Denoising loop
        # We set the index here to remove DtoH sync, helpful especially during compilation.
        # Check out more details here: https://github.com/huggingface/diffusers/pull/11696

        latents = self.denoise(
            latents=latents,
            timesteps=timesteps,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            attention_mask=attention_mask,
            negative_attention_mask=negative_attention_mask,
            use_cfg_guidance=use_cfg_guidance,
            joint_attention_kwargs=attention_kwargs,
            latent_image_ids=latent_image_ids,
            text_ids=text_ids,
            negative_text_ids=negative_text_ids,
            image_embeds=image_embeds,
            negative_image_embeds=negative_image_embeds,
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
