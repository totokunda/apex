import torch
from typing import Dict, Any, Callable, List, Union
from .base import QwenImageBaseEngine
import numpy as np
from PIL import Image
from diffusers.models.controlnets.controlnet_qwenimage import QwenImageControlNetModel, QwenImageMultiControlNetModel

class QwenImageControlNetEngine(QwenImageBaseEngine):
    """QwenImage ControlNet Engine Implementation"""

    def run(
        self,
        prompt: List[str] | str,
        negative_prompt: List[str] | str = None,
        control_guidance_start: Union[float, List[float]] = 0.0,
        control_guidance_end: Union[float, List[float]] = 1.0,
        control_image: str | Image.Image | np.ndarray = None,
        controlnet_conditioning_scale: Union[float, List[float]] = 1.0,
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
        attention_kwargs: Dict[str, Any] = {},
        **kwargs,
    ):
        
        

        if not self.text_encoder:
            self.load_component_by_type("text_encoder")

        self.to_device(self.text_encoder)

        prompt_embeds, prompt_embeds_mask = self.encode_prompt(
            prompt,
            num_images_per_prompt=num_images,
            text_encoder_kwargs=text_encoder_kwargs,
        )
        
        batch_size = prompt_embeds.shape[0]

        if negative_prompt is not None and use_cfg_guidance:
            negative_prompt_embeds, negative_prompt_embeds_mask = self.encode_prompt(
                negative_prompt,
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

        if not self.transformer:
            self.load_component_by_type("transformer")

        self.to_device(self.transformer)

        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(
                self.device, dtype=transformer_dtype
            )
            negative_prompt_embeds_mask = negative_prompt_embeds_mask.to(self.device)
            
        controlnet = self.helpers['controlnet']
        self.to_device(controlnet)
        
        
        if isinstance(controlnet, QwenImageControlNetModel):
            control_image = self.prepare_control_image(control_image, batch_size, height, width, transformer_dtype, use_cfg_guidance)
            control_image_latents = self.vae_encode(control_image, offload=offload)
            control_image_latents = control_image_latents.permute(0, 2, 1, 3, 4)
            control_image = self._pack_latents(
                control_image_latents,
                batch_size=control_image_latents.shape[0],
                num_channels_latents=self.num_channels_latents,
                height=control_image_latents.shape[3],
                width=control_image_latents.shape[4],
            ).to(dtype=prompt_embeds.dtype, device=self.device)
        elif isinstance(controlnet, QwenImageMultiControlNetModel):
            control_images = []
            for control_image_ in control_image:
                control_image_ = self.prepare_control_image(control_image_, batch_size, height, width, transformer_dtype, use_cfg_guidance)
                control_image_latents = self.vae_encode(control_image_, offload=offload)
                control_image_latents = control_image_latents.permute(0, 2, 1, 3, 4)
                control_image_ = self._pack_latents(
                    control_image_latents,
                    batch_size=control_image_latents.shape[0],
                    num_channels_latents=self.num_channels_latents,
                    height=control_image_latents.shape[3],
                    width=control_image_latents.shape[4],
                ).to(dtype=prompt_embeds.dtype, device=self.device)
                control_images.append(control_image_)
            control_image = control_images
        else:
            raise ValueError(f"Unsupported controlnet type: {type(controlnet)}")

        latents = self._get_latents(
            batch_size=num_images,
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
                (
                    1,
                    height // self.vae_scale_factor // 2,
                    width // self.vae_scale_factor // 2,
                )
            ]
        ] * num_images

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
        
        controlnet_keep = []
        for i in range(len(timesteps)):
            keeps = [
                1.0 - float(i / len(timesteps) < s or (i + 1) / len(timesteps) > e)
                for s, e in zip(control_guidance_start, control_guidance_end)
            ]
            controlnet_keep.append(keeps[0] if isinstance(controlnet, QwenImageControlNetModel) else keeps)

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
            controlnet_keep=controlnet_keep,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            control_image=control_image,
            controlnet=controlnet,
        )

        if offload:
            self._offload(self.transformer)
            self._offload(controlnet)

        if return_latents:
            return latents

        latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
        tensor_image = self.vae_decode(latents, offload=offload)[:, :, 0]
        image = self._tensor_to_frame(tensor_image)
        return [image]
