import torch
import math
from typing import Dict, Any, Callable, List
from .base import QwenImageBaseEngine
import numpy as np
from PIL import Image   

class QwenImageEditEngine(QwenImageBaseEngine):
    """QwenImage Edit Engine Implementation"""

    def run(
        self,
        image: Image.Image | np.ndarray | torch.Tensor | str,
        prompt: List[str] | str,
        negative_prompt: List[str] | str = None,
        height: int | None = None,
        width: int | None = None,
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
        attention_kwargs: Dict[str, Any] = {},
        **kwargs,
    ):

        if not self.text_encoder:
            self.load_component_by_type("text_encoder")

        self.to_device(self.text_encoder)
        
        batch_size = (len(prompt) if isinstance(prompt, list) else 1) * num_images
        loaded_image = self._load_image(image)
        loaded_image, calculated_height, calculated_width = self._aspect_ratio_resize(loaded_image, max_area=math.prod(loaded_image.size), mod_value=32)
        preprocessed_image = self.image_processor.preprocess(loaded_image).unsqueeze(2)
        if height is None:
            height = calculated_height
        if width is None:
            width = calculated_width
        
        prompt_embeds, prompt_embeds_mask = self.encode_prompt(
            prompt,
            image=loaded_image,
            device=self.device,
            num_images_per_prompt=num_images,
            text_encoder_kwargs=text_encoder_kwargs,
        )
      

        if negative_prompt is not None and use_cfg_guidance:
            negative_prompt_embeds, negative_prompt_embeds_mask = self.encode_prompt(
                negative_prompt,
                image=loaded_image,
                device=self.device,
                num_images_per_prompt=num_images,
                text_encoder_kwargs=text_encoder_kwargs,
            )
        else:
            negative_prompt_embeds = None
            negative_prompt_embeds_mask = None
        

        if offload:
            self._offload(self.text_encoder)

        transformer_dtype = self.component_dtypes["transformer"]
        prompt_embeds = prompt_embeds.to(self.device, dtype=transformer_dtype)
        prompt_embeds_mask = prompt_embeds_mask.to(self.device)
        
        image_latents = self.vae_encode(preprocessed_image, offload=offload)
        image_latents = torch.cat([image_latents] * batch_size, dim=0)
        image_latent_height, image_latent_width = image_latents.shape[3:]
        image_latents = self._pack_latents(
                image_latents, batch_size, self.num_channels_latents, image_latent_height, image_latent_width
            )
    

        if not self.transformer:
            self.load_component_by_type("transformer")

        self.to_device(self.transformer)
        

        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(
                self.device, dtype=transformer_dtype
            )
            negative_prompt_embeds_mask = negative_prompt_embeds_mask.to(self.device)

        latents = self._get_latents(
            batch_size=batch_size,
            num_channels_latents=self.num_channels_latents,
            height=height,
            width=width,
            dtype=transformer_dtype,
            device=self.device,
            seed=seed,
            generator=generator,
        )

        img_shapes = [
            [
                (1, height // self.vae_scale_factor // 2, width // self.vae_scale_factor // 2),
                (1, calculated_height // self.vae_scale_factor // 2, calculated_width // self.vae_scale_factor // 2),
            ]
        ] * batch_size

        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        image_seq_len = latents.shape[1]

        if not self.scheduler:
            self.load_component_by_type("scheduler")
        self.to_device(self.scheduler)

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
            mu=mu,
            timesteps=timesteps,
        )

        num_warmup_steps = max(
            len(timesteps) - num_inference_steps * self.scheduler.order, 0
        )

        # handle guidance
        if self.transformer.config.guidance_embeds:
            guidance = torch.full(
                [1], guidance_scale, device=self.device, dtype=torch.float32
            )
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None

        txt_seq_lens = (
            prompt_embeds_mask.sum(dim=1).tolist()
            if prompt_embeds_mask is not None
            else None
        )
        negative_txt_seq_lens = (
            negative_prompt_embeds_mask.sum(dim=1).tolist()
            if negative_prompt_embeds_mask is not None
            else None
        )

        self.scheduler.set_begin_index(0)

        latents = self.denoise(
            latents=latents,
            timesteps=timesteps,
            num_inference_steps=num_inference_steps,
            num_warmup_steps=num_warmup_steps,
            guidance=guidance,
            true_cfg_scale=true_cfg_scale,
            image_latents=image_latents,
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_prompt_embeds_mask=negative_prompt_embeds_mask,
            txt_seq_lens=txt_seq_lens,
            negative_txt_seq_lens=negative_txt_seq_lens,
            img_shapes=img_shapes,
            attention_kwargs=attention_kwargs,
            render_on_step=render_on_step,
            render_on_step_callback=render_on_step_callback,
            use_cfg_guidance=use_cfg_guidance,
        )

        if offload:
            self._offload(self.transformer)

        if return_latents:
            return latents

        latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
        tensor_image = self.vae_decode(latents, offload=offload)[:, :, 0]
        image = self._tensor_to_frame(tensor_image)
        return [image]
