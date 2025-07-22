import torch
from typing import Dict, Any, Callable, List, Union, Optional
from PIL import Image
import numpy as np

from .base import WanBaseEngine


class WanApexFramepackEngine(WanBaseEngine):
    """WAN Apex Framepack Engine Implementation"""

    def run(
        self,
        prompt: List[str] | str,
        negative_prompt: List[str] | str = None,
        height: int = 480,
        width: int = 832,
        duration: int | str = 16,
        num_inference_steps: int = 30,
        num_videos: int = 1,
        seed: int | None = None,
        fps: int = 16,
        guidance_scale: float = 5.0,
        use_cfg_guidance: bool = True,
        return_latents: bool = False,
        text_encoder_kwargs: Dict[str, Any] = {},
        attention_kwargs: Dict[str, Any] = {},
        reverse: bool = False,
        render_on_step_callback: Callable = None,
        offload: bool = True,
        render_on_step: bool = False,
        generator: torch.Generator | None = None,
        timesteps: List[int] | None = None,
        timesteps_as_indices: bool = True,
        **kwargs,
    ):

        if not self.text_encoder:
            self.load_component_by_type("text_encoder")

        self.to_device(self.text_encoder)

        prompt_embeds = self.text_encoder.encode(
            prompt,
            device=self.device,
            num_videos_per_prompt=num_videos,
            **text_encoder_kwargs,
        )
        if negative_prompt is not None and use_cfg_guidance:
            negative_prompt_embeds = self.text_encoder.encode(
                negative_prompt,
                device=self.device,
                num_videos_per_prompt=num_videos,
                **text_encoder_kwargs,
            )
        else:
            negative_prompt_embeds = None

        if offload:
            self._offload(self.text_encoder)

        if not self.transformer:
            self.load_component_by_type("transformer")

        self.to_device(self.transformer)
        transformer_dtype = self.component_dtypes["transformer"]
        prompt_embeds = prompt_embeds.to(self.device, dtype=transformer_dtype)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(
                self.device, dtype=transformer_dtype
            )

        framepack_schedule = self.transformer.framepack_schedule

        if not self.scheduler:
            self.load_component_by_type("scheduler")
        self.to_device(self.scheduler)
        scheduler = self.scheduler

        scheduler.set_timesteps(
            num_inference_steps if timesteps is None else 1000, device=self.device
        )
        timesteps, num_inference_steps = self._get_timesteps(
            scheduler=scheduler,
            timesteps=timesteps,
            timesteps_as_indices=timesteps_as_indices,
            num_inference_steps=num_inference_steps,
        )

        num_frames = self._parse_num_frames(duration, fps)

        latents = self._get_latents(
            height,
            width,
            duration,
            fps=fps,
            num_videos=num_videos,
            seed=seed,
            dtype=torch.float32,
            generator=generator,
        )

        total_latent_frames = latents.shape[2]

        denoised_mask = torch.zeros(
            total_latent_frames, dtype=torch.bool, device=self.device
        )
        sections_to_denoise = framepack_schedule.num_sections(num_frames)

        if seed is not None:
            seeds = [seed] * sections_to_denoise
        elif generator is not None:
            seeds = [generator.seed() for _ in range(sections_to_denoise)]
        else:
            seeds = None

        with self._progress_bar(
            total=sections_to_denoise, desc="Denoising sections"
        ) as pbar:
            for section_idx in range(sections_to_denoise):
                (
                    past_latents,
                    past_indices,
                    future_latents,
                    future_indices,
                    target_latents,
                    target_indices,
                ) = framepack_schedule.get_inference_inputs(
                    latents, denoised_mask, reverse=reverse, seeds=seeds
                )

                latent_context = self.transformer.get_latent_context(
                    past_latents,
                    past_indices,
                    future_latents,
                    future_indices,
                    total_latent_frames,
                )

                denoised_latents = self.denoise(
                    timesteps=timesteps,
                    latents=target_latents,
                    transformer_kwargs=dict(
                        encoder_hidden_states=prompt_embeds,
                        attention_kwargs=attention_kwargs,
                        latent_context=latent_context,
                        indices=target_indices,
                    ),
                    unconditional_transformer_kwargs=(
                        dict(
                            encoder_hidden_states=negative_prompt_embeds,
                            attention_kwargs=attention_kwargs,
                            latent_context=latent_context,
                            indices=target_indices,
                        )
                        if negative_prompt_embeds is not None
                        else None
                    ),
                    transformer_dtype=transformer_dtype,
                    use_cfg_guidance=use_cfg_guidance,
                    render_on_step=render_on_step,
                    render_on_step_callback=render_on_step_callback,
                    scheduler=scheduler,
                    guidance_scale=guidance_scale,
                )

                denoised_mask[target_indices] = True
                latents[:, :, target_indices, :, :] = denoised_latents.to(
                    latents.device, dtype=latents.dtype
                )

                self.logger.info(
                    f"Section {section_idx} denoised frames: {', '.join(str(i) for i in target_indices.cpu().tolist())}"
                )

        if offload:
            self._offload(self.transformer)

        if return_latents:
            return latents
        else:
            video = self.vae_decode(latents, offload=offload)
            postprocessed_video = self._postprocess(video)
            return postprocessed_video
