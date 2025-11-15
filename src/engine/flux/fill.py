import torch
from typing import Dict, Any, Callable, List, Union, Optional
from PIL import Image
import numpy as np
from .shared import FluxShared
from diffusers.image_processor import VaeImageProcessor
from src.utils.progress import safe_emit_progress, make_mapped_progress


class FluxFillEngine(FluxShared):
    """Flux Fill Engine Implementation"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_channels_latents = (
            self.transformer.config.in_channels // 4 if self.transformer else 16
        )

    def run(
        self,
        image: Image.Image | str | np.ndarray | torch.Tensor,
        prompt: List[str] | str,
        prompt_2: List[str] | str = None,
        negative_prompt: List[str] | str = None,
        negative_prompt_2: List[str] | str = None,
        mask_image: Optional[Image.Image | str | np.ndarray | torch.Tensor] = None,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 30,
        num_images: int = 1,
        seed: int | None = None,
        true_cfg_scale: float = 1.0,
        guidance_scale: float = 30.0,
        return_latents: bool = False,
        text_encoder_kwargs: Dict[str, Any] = {},
        text_encoder_2_kwargs: Dict[str, Any] = {},
        joint_attention_kwargs: Dict[str, Any] = {},
        render_on_step_callback: Callable = None,
        render_on_step_interval: int = 3,
        progress_callback: Callable = None,
        offload: bool = True,
        render_on_step: bool = False,
        generator: torch.Generator | None = None,
        sigmas: List[float] | None = None,
        timesteps: List[int] | None = None,
        strength: float = 1.0,
        **kwargs,
    ):

        safe_emit_progress(progress_callback, 0.0, "Starting fill pipeline")
        safe_emit_progress(progress_callback, 0.05, "Preparing inputs and RNG")
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
            
        mask_processor = VaeImageProcessor(
            vae_scale_factor=self.image_processor.config.vae_scale_factor,
            vae_latent_channels=self.image_processor.config.vae_latent_channels,
            do_normalize=False,
            do_binarize=True,
            do_convert_grayscale=True,
        )
            
        use_cfg_guidance = true_cfg_scale > 1.0 and negative_prompt is not None
        
        safe_emit_progress(progress_callback, 0.10, "Encoding prompts")
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
            self._offload(self.text_encoder)
            self._offload(self.text_encoder_2)

        transformer_dtype = self.component_dtypes.get("transformer", None)
        
        safe_emit_progress(progress_callback, 0.15, "Loading and preprocessing images")
        image = self._load_image(image)
        init_image = self.image_processor.preprocess(image, height=height, width=width)
        init_image = init_image.to(dtype=torch.float32)
        
        image_seq_len = (int(height) // self.vae_scale_factor // 2) * (int(width) // self.vae_scale_factor // 2)
        sigmas = (
            np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
            if sigmas is None
            else sigmas
        )
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
        
        safe_emit_progress(progress_callback, 0.25, "Preparing timesteps")
        timesteps, num_inference_steps = self._get_timesteps(
            self.scheduler,
            num_inference_steps,
            sigmas=sigmas,
            timesteps=timesteps,
            mu=mu,
            strength=strength,
        )

        latent_timestep = timesteps[:1].repeat(prompt_embeds.shape[0])
        batch_size = prompt_embeds.shape[0]

        safe_emit_progress(progress_callback, 0.30, "Initializing latents")
        latents, latent_ids = self._get_latents(
            image=init_image,
            batch_size=batch_size,
            num_channels_latents=self.num_channels_latents,
            height=height,
            width=width,
            dtype=transformer_dtype,
            device=self.device,
            generator=generator,
            timestep=latent_timestep,
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
        
        safe_emit_progress(progress_callback, 0.35, "Preparing mask and masked image")
        mask_image = self._load_image(mask_image)
        
        mask_image = mask_processor.preprocess(mask_image, height=height, width=width)

        masked_image = init_image * (1 - mask_image)
        
        masked_image = masked_image.to(device=self.device, dtype=transformer_dtype)


        height, width = init_image.shape[-2:]
        mask, masked_image_latents = self.prepare_mask_latents(
            mask_image,
            masked_image,
            latents.shape[0],
            self.num_channels_latents,
            num_images,
            height,
            width,
            transformer_dtype,
            self.device,
            offload,
        )
        
        masked_image_latents = torch.cat((masked_image_latents, mask), dim=-1)

        # Reserve a progress gap for denoising [0.50, 0.90]
        denoise_progress_callback = make_mapped_progress(progress_callback, 0.50, 0.90)
        # Set preview context for per-step rendering on the main engine (denoise runs there)

        self._preview_height = height
        self._preview_width = width
        self._preview_offload = offload

        safe_emit_progress(progress_callback, 0.50, "Starting denoise")
        latents = self.denoise(
            latents=latents,
            timesteps=timesteps,
            num_inference_steps=num_inference_steps,
            concat_latents=masked_image_latents,
            guidance=guidance,
            prompt_embeds=prompt_embeds,
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
            denoise_progress_callback=denoise_progress_callback,
            render_on_step_interval=render_on_step_interval,
        )

        if return_latents:
            safe_emit_progress(progress_callback, 1.0, "Returning latents")
            return latents

        latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
        image = self.vae_decode(latents, offload=offload)
        image = self._tensor_to_frame(image)
        safe_emit_progress(progress_callback, 1.0, "Completed fill pipeline")
        return image
