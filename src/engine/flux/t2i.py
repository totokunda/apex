import torch
from typing import Dict, Any, Callable, List, Union, Optional
from PIL import Image
import numpy as np
from .base import FluxBaseEngine


class FluxT2IEngine(FluxBaseEngine):
    """Flux Text-to-Image Engine Implementation"""

    def run(
        self,
        prompt: List[str] | str,
        prompt_2: List[str] | str = None,
        negative_prompt: List[str] | str = None,
        negative_prompt_2: List[str] | str = None,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 30,
        num_images: int = 1,
        seed: int | None = None,
        true_cfg_scale: float = 1.0,
        guidance_scale: float = 3.5,
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
        ip_adapter_image: Optional[Image.Image | str | np.ndarray] = None,
        ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        negative_ip_adapter_image: Optional[Image.Image | str | np.ndarray] = None,
        negative_ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        **kwargs,
    ):

        if not hasattr(self, "text_encoder") or not self.text_encoder:
            self.load_component_by_name("text_encoder")

        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        self.to_device(self.text_encoder)

        pooled_prompt_embeds = self.text_encoder.encode(
            f"{prompt}",
            device=self.device,
            num_videos_per_prompt=num_images,
            output_type="pooler_output",
            **text_encoder_kwargs,
        )

        use_cfg_guidance = true_cfg_scale > 1.0 and negative_prompt is not None

        if negative_prompt is not None and use_cfg_guidance:
            negative_pooled_prompt_embeds = self.text_encoder.encode(
                negative_prompt,
                device=self.device,
                num_videos_per_prompt=num_images,
                output_type="pooler_output",
                **text_encoder_kwargs,
            )
        else:
            negative_pooled_prompt_embeds = None

        if offload:
            self._offload(self.text_encoder)

        if not hasattr(self, "text_encoder_2") or not self.text_encoder_2:
            self.load_component_by_name("text_encoder_2")

        self.to_device(self.text_encoder_2)

        if not prompt_2:
            prompt_2 = prompt

        if not negative_prompt_2:
            negative_prompt_2 = negative_prompt

        prompt_embeds = self.text_encoder_2.encode(
            prompt_2,
            device=self.device,
            num_videos_per_prompt=num_images,
            output_type="hidden_states",
            **text_encoder_2_kwargs,
        )

        text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(
            device=self.device, dtype=prompt_embeds.dtype
        )

        if negative_prompt_2 is not None and use_cfg_guidance:
            negative_prompt_embeds = self.text_encoder_2.encode(
                negative_prompt_2,
                device=self.device,
                num_videos_per_prompt=num_images,
                output_type="hidden_states",
                **text_encoder_2_kwargs,
            )
            negative_text_ids = torch.zeros(negative_prompt_embeds.shape[1], 3).to(
                device=self.device, dtype=negative_prompt_embeds.dtype
            )
        else:
            negative_prompt_embeds = None
            negative_text_ids = None

        if offload:
            self._offload(self.text_encoder_2)

        transformer_dtype = self.component_dtypes.get("transformer", None)

        latents, latent_image_ids = self._get_latents(
            batch_size=num_images,
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
        if (
            hasattr(self.scheduler.config, "use_flow_sigmas")
            and self.scheduler.config.use_flow_sigmas
        ):
            sigmas = None
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

        if (ip_adapter_image is not None or ip_adapter_image_embeds is not None) and (
            negative_ip_adapter_image is None
            and negative_ip_adapter_image_embeds is None
        ):
            negative_ip_adapter_image = np.zeros((width, height, 3), dtype=np.uint8)
            negative_ip_adapter_image = [
                negative_ip_adapter_image
            ] * self.transformer.encoder_hid_proj.num_ip_adapters

        elif (ip_adapter_image is None and ip_adapter_image_embeds is None) and (
            negative_ip_adapter_image is not None
            or negative_ip_adapter_image_embeds is not None
        ):
            ip_adapter_image = np.zeros((width, height, 3), dtype=np.uint8)
            ip_adapter_image = [
                ip_adapter_image
            ] * self.transformer.encoder_hid_proj.num_ip_adapters

        image_embeds = None
        negative_image_embeds = None
        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                self.device,
                num_images,
            )
        if (
            negative_ip_adapter_image is not None
            or negative_ip_adapter_image_embeds is not None
        ):
            negative_image_embeds = self.prepare_ip_adapter_image_embeds(
                negative_ip_adapter_image,
                negative_ip_adapter_image_embeds,
                self.device,
                num_images,
            )

        # 6. Denoising loop
        # We set the index here to remove DtoH sync, helpful especially during compilation.
        # Check out more details here: https://github.com/huggingface/diffusers/pull/11696
        self.scheduler.set_begin_index(0)

        latents = self.denoise(
            latents=latents,
            timesteps=timesteps,
            num_inference_steps=num_inference_steps,
            guidance=guidance,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            use_cfg_guidance=use_cfg_guidance,
            joint_attention_kwargs=joint_attention_kwargs,
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
