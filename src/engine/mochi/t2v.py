import torch
from typing import Dict, Any, Callable, List, Union, Optional
from PIL import Image
import numpy as np
from .base import MochiBaseEngine
from src.ui.nodes import UINode


def linear_quadratic_schedule(num_steps, threshold_noise=0.025, linear_steps=None):
    if linear_steps is None:
        linear_steps = num_steps // 2
    if num_steps < 2:
        return np.array([1.0])
    linear_sigma_schedule = [
        i * threshold_noise / linear_steps for i in range(linear_steps)
    ]
    threshold_noise_step_diff = linear_steps - threshold_noise * num_steps
    quadratic_steps = num_steps - linear_steps
    quadratic_coef = threshold_noise_step_diff / (linear_steps * quadratic_steps**2)
    linear_coef = threshold_noise / linear_steps - 2 * threshold_noise_step_diff / (
        quadratic_steps**2
    )
    const = quadratic_coef * (linear_steps**2)
    quadratic_sigma_schedule = [
        quadratic_coef * (i**2) + linear_coef * i + const
        for i in range(linear_steps, num_steps)
    ]
    sigma_schedule = linear_sigma_schedule + quadratic_sigma_schedule
    sigma_schedule = [1.0 - x for x in sigma_schedule]
    return np.array(sigma_schedule)


class MochiT2VEngine(MochiBaseEngine):
    """Mochi Text-to-Video Engine Implementation"""

    def run(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: int = 480,
        width: int = 848,
        num_frames: int = 19,
        num_inference_steps: int = 28,
        guidance_scale: float = 3.5,
        num_videos_per_prompt: int = 1,
        seed: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.Tensor] = None,
        output_type: str = "pil",
        return_latents: bool = False,
        offload: bool = True,
        render_on_step: bool = False,
        render_on_step_callback: Optional[Callable] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        max_sequence_length: int = 256,
        input_nodes: List[UINode] = None,
        **kwargs,
    ):
        default_kwargs = self._get_default_kwargs("run")
        preprocessed_kwargs = self._preprocess_kwargs(input_nodes, **kwargs)
        final_kwargs = {**default_kwargs, **preprocessed_kwargs}

        height = final_kwargs.get("height", height)
        width = final_kwargs.get("width", width)
        num_frames = final_kwargs.get("num_frames", num_frames)
        num_inference_steps = final_kwargs.get(
            "num_inference_steps", num_inference_steps
        )
        guidance_scale = final_kwargs.get("guidance_scale", guidance_scale)
        num_videos_per_prompt = final_kwargs.get(
            "num_videos_per_prompt", num_videos_per_prompt
        )
        seed = final_kwargs.get("seed", seed)
        generator = final_kwargs.get("generator", generator)
        latents = final_kwargs.get("latents", latents)
        output_type = final_kwargs.get("output_type", output_type)
        return_latents = final_kwargs.get("return_latents", return_latents)
        offload = final_kwargs.get("offload", offload)
        render_on_step = final_kwargs.get("render_on_step", render_on_step)
        render_on_step_callback = final_kwargs.get(
            "render_on_step_callback", render_on_step_callback
        )
        attention_kwargs = final_kwargs.get("attention_kwargs", attention_kwargs)
        max_sequence_length = final_kwargs.get(
            "max_sequence_length", max_sequence_length
        )
        prompt = final_kwargs.get("prompt", prompt)
        negative_prompt = final_kwargs.get("negative_prompt", negative_prompt)

        if not self.text_encoder:
            self.load_component_by_type("text_encoder")
        self.to_device(self.text_encoder)

        prompt_embeds, prompt_attention_mask = self.text_encoder.encode(
            prompt,
            device=self.device,
            num_videos_per_prompt=num_videos_per_prompt,
            max_sequence_length=max_sequence_length,
            return_attention_mask=True,
        )

        use_cfg_guidance = guidance_scale > 1.0
        if use_cfg_guidance:
            if negative_prompt is None:
                negative_prompt = ""
            negative_prompt_embeds, negative_prompt_attention_mask = (
                self.text_encoder.encode(
                    negative_prompt,
                    device=self.device,
                    num_videos_per_prompt=num_videos_per_prompt,
                    max_sequence_length=max_sequence_length,
                    return_attention_mask=True,
                )
            )
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            prompt_attention_mask = torch.cat(
                [negative_prompt_attention_mask, prompt_attention_mask], dim=0
            )

        if offload:
            self._offload(self.text_encoder)

        if not self.transformer:
            self.load_component_by_type("transformer")
        self.to_device(self.transformer)

        if not self.scheduler:
            self.load_component_by_type("scheduler")
        self.to_device(self.scheduler)

        if generator is None and seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        if latents is None:
            latents = self.prepare_latents(
                num_videos_per_prompt,
                self.num_channels_latents,
                height,
                width,
                num_frames,
                prompt_embeds.dtype,
                self.device,
                generator,
            )

        threshold_noise = 0.025
        sigmas = linear_quadratic_schedule(num_inference_steps, threshold_noise)

        timesteps, num_inference_steps = self._get_timesteps(
            num_inference_steps=num_inference_steps,
            device=self.device,
            sigmas=sigmas,
        )
        num_warmup_steps = max(
            len(timesteps) - num_inference_steps * self.scheduler.order, 0
        )

        latents = self.denoise(
            latents=latents,
            timesteps=timesteps,
            num_warmup_steps=num_warmup_steps,
            num_inference_steps=num_inference_steps,
            prompt_embeds=prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            attention_kwargs=attention_kwargs,
            use_cfg_guidance=use_cfg_guidance,
            guidance_scale=guidance_scale,
            render_on_step=render_on_step,
            scheduler=self.scheduler,
            transformer_dtype=self.component_dtypes.get("transformer", torch.float16),
            render_on_step_callback=render_on_step_callback,
        )

        if offload:
            self._offload(self.transformer)

        if return_latents:
            return latents

        video = self.vae_decode(latents, offload=offload, dtype=prompt_embeds.dtype)
        video = self._postprocess(video, output_type=output_type)
        return video
