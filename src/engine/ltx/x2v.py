from .base import LTXBaseEngine, LTXVideoCondition
from typing import Dict, Any, Callable, List, Union, Optional
import torch
import numpy as np
from PIL import Image
import math
from src.attention.processors.ltx_processor import SkipLayerStrategy
from src.scheduler.rf import TimestepShifter


class LTXX2VEngine(LTXBaseEngine):
    """LTX Text-to-Video Engine Implementation"""

    def run(
        self,
        prompt: List[str] | str,
        negative_prompt: List[str] | str = None,
        conditions: List[LTXVideoCondition] | LTXVideoCondition | None = None,
        initial_latents: Optional[torch.Tensor] = None,
        initial_image: Optional[Image.Image] = None,
        initial_video: Optional[List[Image.Image] | torch.Tensor] = None,
        height: int = 480,
        width: int = 832,
        duration: int | str = 25,
        num_inference_steps: int = 30,
        skip_initial_inference_steps: int = 0,
        skip_final_inference_steps: int = 0,
        num_videos: int = 1,
        seed: int | None = None,
        eta: float = 0.0,
        fps: int = 30,
        text_encoder_kwargs: Dict[str, Any] = {},
        guidance_scale: float | List[float] = 3.0,
        stg_scale: float | List[float] = 1.0,
        rescaling_scale: float | List[float] = 1.0,
        image_cond_noise_scale: float = 0.15,
        offload: bool = True,
        render_on_step: bool = False,
        generator: torch.Generator | None = None,
        timesteps: List[int] | None = None,
        guidance_timesteps: List[int] | None = None,
        render_on_step_callback: Callable = None,
        cfg_star_rescale: bool = False,
        return_latents: bool = False,
        decode_timestep: Union[float, List[float]] = 0.0,
        decode_noise_scale: Optional[Union[float, List[float]]] = None,
        stochastic_sampling: bool = False,
        tone_map_compression_ratio: float = 0.0,
        skip_block_list: Optional[List[List[int]]] = None,
        skip_layer_strategy: Optional[SkipLayerStrategy] | Optional[str] = None,
        **kwargs,
    ):

        if not self.text_encoder:
            self.load_component_by_type("text_encoder")

        self.to_device(self.text_encoder)

        if skip_layer_strategy is not None:
            if isinstance(skip_layer_strategy, str):
                skip_layer_strategy = SkipLayerStrategy(skip_layer_strategy)

        prompt_embeds, prompt_attention_mask = self.text_encoder.encode(
            prompt,
            device=self.device,
            num_videos_per_prompt=num_videos,
            return_attention_mask=True,
            **text_encoder_kwargs,
        )

        if negative_prompt:
            negative_prompt_embeds, negative_prompt_attention_mask = (
                self.text_encoder.encode(
                    negative_prompt,
                    device=self.device,
                    num_videos_per_prompt=num_videos,
                    return_attention_mask=True,
                    **text_encoder_kwargs,
                )
            )
        else:
            negative_prompt_embeds, negative_prompt_attention_mask = torch.zeros_like(
                prompt_embeds
            ), torch.zeros_like(prompt_attention_mask)

        if offload:
            self._offload(self.text_encoder)

        if not self.scheduler:
            self.load_component_by_type("scheduler")

        # load transformer
        if not self.transformer:
            self.load_component_by_type("transformer")

        self.to_device(self.transformer)

        transformer_dtype = self.component_dtypes["transformer"]

        prompt_embeds_batch = torch.cat(
            [negative_prompt_embeds, prompt_embeds, prompt_embeds], dim=0
        )

        prompt_attention_mask_batch = torch.cat(
            [
                negative_prompt_attention_mask,
                prompt_attention_mask,
                prompt_attention_mask,
            ],
            dim=0,
        )

        if initial_image is not None:
            initial_image = self._load_image(initial_image)
            initial_image, height, width = self._aspect_ratio_resize(
                initial_image,
                max_area=height * width,
                mod_value=self.vae_scale_factor_spatial,
            )
            video_input = self.video_processor.preprocess(
                initial_image, height, width
            ).unsqueeze(2)
        elif initial_video is not None:
            video_input = self._load_video(initial_video)
            for i, frame in enumerate(video_input):
                frame, height, width = self._aspect_ratio_resize(
                    frame,
                    max_area=height * width,
                    mod_value=self.vae_scale_factor_spatial,
                )
                video_input[i] = frame
            video_input = self.video_processor.preprocess_video(
                video_input, height, width
            )
        else:
            video_input = None

        if video_input is not None:
            latent_input = self.vae_encode(
                video_input,
                sample_mode="mode",
                normalize_latents_dtype=torch.float32,
                offload=offload,
            )
            shape = latent_input.shape
        elif initial_latents is not None:
            latent_input = initial_latents
            shape = latent_input.shape
        else:
            latent_input = None
            shape = None

        num_frames = self._parse_num_frames(duration, fps)
        noise = self._get_latents(
            height,
            width,
            num_frames,
            fps,
            num_videos,
            shape=shape,
            dtype=transformer_dtype,
            seed=seed,
            generator=generator,
            parse_frames=(video_input is None and initial_latents is None),
        )

        noise = noise * self.scheduler.init_noise_sigma

        retrieve_timesteps_kwargs = {}
        if isinstance(self.scheduler, TimestepShifter):
            retrieve_timesteps_kwargs["samples_shape"] = noise.shape

        timesteps, num_inference_steps = self._get_timesteps(
            self.scheduler,
            num_inference_steps,
            self.device,
            timesteps,
            skip_initial_inference_steps=skip_initial_inference_steps,
            skip_final_inference_steps=skip_final_inference_steps,
            **retrieve_timesteps_kwargs,
        )

        if latent_input is not None:
            latents = timesteps[0] * noise + (1 - timesteps[0]) * latent_input
        else:
            latents = noise

        latent_height = latents.shape[3]
        latent_width = latents.shape[4]

        if guidance_timesteps:
            guidance_mapping = []
            for timestep in timesteps:
                indices = [
                    i for i, val in enumerate(guidance_timesteps) if val <= timestep
                ]
                # assert len(indices) > 0, f"No guidance timestep found for {timestep}"
                guidance_mapping.append(
                    indices[0] if len(indices) > 0 else (len(guidance_timesteps) - 1)
                )

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.

        if not isinstance(guidance_scale, List):
            guidance_scale = [guidance_scale] * len(timesteps)
        else:
            guidance_scale = [
                guidance_scale[guidance_mapping[i]] for i in range(len(timesteps))
            ]

        if not isinstance(stg_scale, List):
            stg_scale = [stg_scale] * len(timesteps)
        else:
            stg_scale = [stg_scale[guidance_mapping[i]] for i in range(len(timesteps))]

        if not isinstance(rescaling_scale, List):
            rescaling_scale = [rescaling_scale] * len(timesteps)
        else:
            rescaling_scale = [
                rescaling_scale[guidance_mapping[i]] for i in range(len(timesteps))
            ]

        if skip_block_list is not None:
            # Convert single list to list of lists if needed
            if len(skip_block_list) == 0 or not isinstance(skip_block_list[0], list):
                skip_block_list = [skip_block_list] * len(timesteps)
            else:
                new_skip_block_list = []
                for i, timestep in enumerate(timesteps):
                    new_skip_block_list.append(skip_block_list[guidance_mapping[i]])
                skip_block_list = new_skip_block_list

        # patch latents
        patchifier = self.helpers["ltx.patchifier"]
        causal_fix = getattr(
            self.transformer.config, "causal_temporal_positioning", False
        )

        latents, pixel_coords, conditioning_mask, num_cond_latents = (
            self.prepare_conditioning(
                conditioning_items=conditions,
                causal_fix=causal_fix,
                patchifier=patchifier,
                init_latents=latents,
                num_frames=num_frames,
                height=height,
                width=width,
                generator=generator,
            )
        )

        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        num_warmup_steps = max(
            len(timesteps) - num_inference_steps * self.scheduler.order, 0
        )

        latents = self.denoise(
            conditioning_mask=conditioning_mask,
            latents=latents,
            timesteps=timesteps,
            num_inference_steps=num_inference_steps,
            prompt_embeds_batch=prompt_embeds_batch,
            prompt_attention_mask_batch=prompt_attention_mask_batch,
            guidance_scale=guidance_scale,
            stg_scale=stg_scale,
            rescaling_scale=rescaling_scale,
            num_videos=num_videos,
            fps=fps,
            image_cond_noise_scale=image_cond_noise_scale,
            generator=generator,
            stochastic_sampling=stochastic_sampling,
            cfg_star_rescale=cfg_star_rescale,
            render_on_step=render_on_step,
            render_on_step_callback=render_on_step_callback,
            num_warmup_steps=num_warmup_steps,
            extra_step_kwargs=extra_step_kwargs,
            pixel_coords=pixel_coords,
            transformer_dtype=transformer_dtype,
            skip_block_list=skip_block_list,
            skip_layer_strategy=skip_layer_strategy,
        )

        if offload:
            self._offload(self.transformer)

        latents = latents[:, num_cond_latents:]

        latents = patchifier.unpatchify(
            latents=latents,
            output_height=latent_height,
            output_width=latent_width,
            out_channels=self.transformer.in_channels
            // math.prod(patchifier.patch_size),
        )

        if offload:
            self._offload(self.transformer)

        if return_latents:
            return latents

        return self.prepare_output(
            latents=latents,
            offload=offload,
            generator=generator,
            decode_timestep=decode_timestep,
            decode_noise_scale=decode_noise_scale,
            tone_map_compression_ratio=tone_map_compression_ratio,
        )
