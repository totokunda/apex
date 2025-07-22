from .base import LTXBaseEngine
from diffusers.utils.torch_utils import randn_tensor
from typing import Dict, Any, Callable, List, Union, Optional
from PIL import Image
import torch
import numpy as np
from src.mixins import LoaderMixin


def linear_quadratic_schedule(num_steps, threshold_noise=0.025, linear_steps=None):
    if linear_steps is None:
        linear_steps = num_steps // 2
    if num_steps < 2:
        return torch.tensor([1.0])
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
    sigma_schedule = linear_sigma_schedule + quadratic_sigma_schedule + [1.0]
    sigma_schedule = [1.0 - x for x in sigma_schedule]
    return torch.tensor(sigma_schedule[:-1])


class LTXVideoCondition(LoaderMixin):
    def __init__(
        self,
        image: Optional[Image.Image] = None,
        video: Optional[List[Image.Image]] = None,
        frame_index: int = 0,
        strength: float = 1.0,
    ):
        self.image = self._load_image(image) if image is not None else None
        self.video = self._load_video(video) if video is not None else None
        self.frame_index = frame_index
        self.strength = strength
        self.height = (
            self.image.height
            if self.image is not None
            else self.video[0].height if self.video is not None else None
        )
        self.width = (
            self.image.width
            if self.image is not None
            else self.video[0].width if self.video is not None else None
        )


class LTXControlEngine(LTXBaseEngine):
    """LTX Control Engine Implementation"""

    def run(
        self,
        conditions: List[LTXVideoCondition] | LTXVideoCondition | None = None,
        image: (
            Union[
                Image.Image, List[Image.Image], str, List[str], np.ndarray, torch.Tensor
            ]
            | None
        ) = None,
        video: (
            Union[
                List[Image.Image],
                List[List[Image.Image]],
                str,
                List[str],
                np.ndarray,
                torch.Tensor,
            ]
            | None
        ) = None,
        frame_index: Union[int, List[int]] = 0,
        strength: Union[float, List[float]] = 1.0,
        prompt: List[str] | str | None = None,
        negative_prompt: List[str] | str | None = None,
        height: int = 480,
        width: int = 832,
        duration: int | str = 16,
        num_inference_steps: int = 30,
        num_videos: int = 1,
        seed: int | None = None,
        fps: int = 25,
        offload: bool = True,
        render_on_step: bool = False,
        image_cond_noise_scale: float = 0.15,
        render_on_step_callback: Callable = None,
        return_latents: bool = False,
        num_prefix_latent_frames: int = 2,
        timesteps: List[int] | None = None,
        decode_timestep: Union[float, List[float]] = 0.0,
        decode_noise_scale: Optional[Union[float, List[float]]] = None,
        use_cfg_guidance: bool = True,
        text_encoder_kwargs: Dict[str, Any] = {},
        guidance_scale: float = 3.0,
        guidance_rescale: float = 0.0,
        attention_kwargs: Dict[str, Any] = {},
        generator: torch.Generator | None = None,
        **kwargs,
    ):
        postprocessor_kwargs = kwargs.get("postprocessor_kwargs", None)

        if not self.text_encoder:
            self.load_component_by_type("text_encoder")

        self.to_device(self.text_encoder)
        text_encoder_kwargs["max_sequence_length"] = 256
        text_encoder_kwargs["use_mask_in_input"] = True

        prompt_embeds, prompt_attention_mask = self.text_encoder.encode(
            prompt,
            device=self.device,
            num_videos_per_prompt=num_videos,
            return_attention_mask=True,
            **text_encoder_kwargs,
        )

        if negative_prompt is not None and use_cfg_guidance:
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

        if use_cfg_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            prompt_attention_mask = torch.cat(
                [negative_prompt_attention_mask, prompt_attention_mask], dim=0
            )

        if offload:
            self._offload(self.text_encoder)

        if conditions is not None:
            if not isinstance(conditions, list):
                conditions = [conditions]
            strength = [condition.strength for condition in conditions]
            frame_index = [condition.frame_index for condition in conditions]
            image = [condition.image for condition in conditions]
            video = [condition.video for condition in conditions]

        elif image is not None or video is not None:
            if not isinstance(image, list):
                image = [self._load_image(image) if image is not None else None]
                num_conditions = 1
            elif isinstance(image, list):
                image = [self._load_image(img) for img in image if img is not None]
                num_conditions = len(image)
            if not isinstance(video, list):
                video = [self._load_video(video) if video is not None else None]
                num_conditions = 1
            elif isinstance(video, list):
                video = [self._load_video(vid) for vid in video if vid is not None]
                num_conditions = len(video)
            if not isinstance(frame_index, list):
                frame_index = [frame_index] * num_conditions
            if not isinstance(strength, list):
                strength = [strength] * num_conditions

        num_frames = self._parse_num_frames(duration, fps)
        latent_num_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        latent_height = height // self.vae_scale_factor_spatial
        latent_width = width // self.vae_scale_factor_spatial
        batch_size = num_videos
        height = height // self.vae_scale_factor_spatial * self.vae_scale_factor_spatial
        width = width // self.vae_scale_factor_spatial * self.vae_scale_factor_spatial

        conditioning_tensors = []
        is_conditioning_image_or_video = image is not None or video is not None

        if is_conditioning_image_or_video:
            for (
                condition_image,
                condition_video,
                condition_frame_index,
            ) in zip(image, video, frame_index):
                if condition_image is not None:
                    condition_tensor = (
                        self.video_processor.preprocess(condition_image, height, width)
                        .unsqueeze(2)
                        .to(device=self.device)
                    )
                elif condition_video is not None:
                    condition_tensor = self.video_processor.preprocess_video(
                        condition_video, height, width
                    )
                    num_frames_input = condition_tensor.size(2)
                    num_frames_output = self.trim_conditioning_sequence(
                        condition_frame_index, num_frames_input, num_frames
                    )
                    condition_tensor = condition_tensor[:, :, :num_frames_output]
                    condition_tensor = condition_tensor.to(device=self.device)
                else:
                    raise ValueError(
                        "Either `image` or `video` must be provided for conditioning."
                    )

                if condition_tensor.size(2) % self.vae_scale_factor_temporal != 1:
                    raise ValueError(
                        f"Number of frames in the video must be of the form (k * {self.vae_scale_factor_temporal} + 1) "
                        f"but got {condition_tensor.size(2)} frames."
                    )
                conditioning_tensors.append(condition_tensor)

        if not self.transformer:
            self.load_component_by_type("transformer")

        self.to_device(self.transformer)
        transformer_dtype = self.component_dtypes["transformer"]

        if not self.scheduler:
            self.load_component_by_type("scheduler")
        self.to_device(self.scheduler)
        scheduler = self.scheduler

        prompt_embeds = prompt_embeds.to(device=self.device, dtype=transformer_dtype)
        prompt_attention_mask = prompt_attention_mask.to(device=self.device)

        if timesteps is None:
            sigmas = linear_quadratic_schedule(num_inference_steps)
            timesteps = sigmas * 1000
        timesteps, num_inference_steps = self._get_timesteps(
            scheduler=scheduler,
            num_inference_steps=num_inference_steps,
            device=self.device,
            timesteps=timesteps,
            sigmas=sigmas if not timesteps else None,
        )

        sigmas = scheduler.sigmas
        num_warmup_steps = max(
            len(timesteps) - num_inference_steps * scheduler.order, 0
        )

        self._num_timesteps = len(timesteps)

        latents = self._get_latents(
            height=height,
            width=width,
            duration=duration,
            fps=fps,
            seed=seed,
            num_videos=num_videos,
            dtype=torch.float32,
            generator=generator,
        )

        if len(conditioning_tensors) > 0:
            condition_latent_frames_mask = torch.zeros(
                (batch_size, latent_num_frames), device=self.device, dtype=torch.float32
            )

            extra_conditioning_latents = []
            extra_conditioning_video_ids = []
            extra_conditioning_mask = []
            extra_conditioning_num_latents = 0

            for data, strength, frame_index in zip(
                conditioning_tensors, strength, frame_index
            ):
                condition_latents = self.vae_encode(
                    data,
                    sample_generator=generator,
                    offload=offload,
                    sample_mode="sample",
                    dtype=torch.float32,
                )
                num_data_frames = data.size(2)
                num_cond_frames = condition_latents.size(2)
                if frame_index == 0:
                    latents[:, :, :num_cond_frames] = torch.lerp(
                        latents[:, :, :num_cond_frames], condition_latents, strength
                    )
                    condition_latent_frames_mask[:, :num_cond_frames] = strength
                else:
                    if num_data_frames > 1:
                        if num_cond_frames < num_prefix_latent_frames:
                            raise ValueError(
                                f"Number of latent frames must be at least {num_prefix_latent_frames} but got {num_data_frames}."
                            )
                        if num_cond_frames > num_prefix_latent_frames:
                            start_frame = (
                                frame_index // self.vae_scale_factor_temporal
                                + num_prefix_latent_frames
                            )
                            end_frame = (
                                start_frame + num_cond_frames - num_prefix_latent_frames
                            )
                            latents[:, :, start_frame:end_frame] = torch.lerp(
                                latents[:, :, start_frame:end_frame],
                                condition_latents[:, :, num_prefix_latent_frames:],
                                strength,
                            )
                            condition_latent_frames_mask[:, start_frame:end_frame] = (
                                strength
                            )
                            condition_latents = condition_latents[
                                :, :, :num_prefix_latent_frames
                            ]
                    noise = randn_tensor(
                        condition_latents.shape,
                        generator=generator,
                        device=self.device,
                        dtype=transformer_dtype,
                    )
                    condition_latents = torch.lerp(noise, condition_latents, strength)
                    condition_video_ids = self._prepare_video_ids(
                        batch_size,
                        condition_latents.size(2),
                        latent_height,
                        latent_width,
                        patch_size=self.transformer_spatial_patch_size,
                        patch_size_t=self.transformer_temporal_patch_size,
                        device=self.device,
                    )
                    condition_video_ids = self._scale_video_ids(
                        condition_video_ids,
                        scale_factor=self.vae_scale_factor_spatial,
                        scale_factor_t=self.vae_scale_factor_temporal,
                        frame_index=frame_index,
                        device=self.device,
                    )
                    condition_latents = self._pack_latents(
                        condition_latents,
                        self.transformer_spatial_patch_size,
                        self.transformer_temporal_patch_size,
                    )

                    condition_conditioning_mask = torch.full(
                        condition_latents.shape[:2],
                        strength,
                        device=self.device,
                        dtype=transformer_dtype,
                    )
                    extra_conditioning_latents.append(condition_latents)
                    extra_conditioning_video_ids.append(condition_video_ids)
                    extra_conditioning_mask.append(condition_conditioning_mask)
                    extra_conditioning_num_latents += condition_latents.size(1)

        video_ids = self._prepare_video_ids(
            batch_size,
            latent_num_frames,
            latent_height,
            latent_width,
            patch_size_t=self.transformer_temporal_patch_size,
            patch_size=self.transformer_spatial_patch_size,
            device=self.device,
        )

        if len(conditioning_tensors) > 0:
            conditioning_mask = condition_latent_frames_mask.gather(1, video_ids[:, 0])
        else:
            conditioning_mask, extra_conditioning_num_latents = None, 0

        video_ids = self._scale_video_ids(
            video_ids,
            scale_factor=self.vae_scale_factor_spatial,
            scale_factor_t=self.vae_scale_factor_temporal,
            frame_index=0,
            device=self.device,
        )

        latents = self._pack_latents(
            latents,
            self.transformer_spatial_patch_size,
            self.transformer_temporal_patch_size,
        )

        if len(conditioning_tensors) > 0 and len(extra_conditioning_latents) > 0:
            latents = torch.cat([*extra_conditioning_latents, latents], dim=1)
            video_ids = torch.cat([*extra_conditioning_video_ids, video_ids], dim=2)
            conditioning_mask = torch.cat(
                [*extra_conditioning_mask, conditioning_mask], dim=1
            )

        video_ids = video_ids.float()
        video_ids[:, 0] = video_ids[:, 0] * (1.0 / fps)

        if use_cfg_guidance:
            video_ids = torch.cat([video_ids, video_ids], dim=0)

        init_latents = latents.clone() if is_conditioning_image_or_video else None

        if len(conditioning_tensors) > 0 and len(extra_conditioning_latents) > 0:
            latents = torch.cat([*extra_conditioning_latents, latents], dim=1)
            video_ids = torch.cat([*extra_conditioning_video_ids, video_ids], dim=2)
            conditioning_mask = torch.cat(
                [*extra_conditioning_mask, conditioning_mask], dim=1
            )

        latents = self.denoise(
            timesteps=timesteps,
            latents=latents,
            transformer_dtype=transformer_dtype,
            use_cfg_guidance=use_cfg_guidance,
            guidance_scale=guidance_scale,
            guidance_rescale=guidance_rescale,
            num_inference_steps=num_inference_steps,
            prompt_embeds=prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            latent_num_frames=latent_num_frames,
            latent_height=latent_height,
            latent_width=latent_width,
            attention_kwargs=attention_kwargs,
            render_on_step=render_on_step,
            render_on_step_callback=render_on_step_callback,
            conditioning_mask=conditioning_mask,
            is_conditioning_image_or_video=is_conditioning_image_or_video,
            generator=generator,
            init_latents=init_latents,
            video_coords=video_ids,
            scheduler=scheduler,
            image_cond_noise_scale=image_cond_noise_scale,
            num_warmup_steps=num_warmup_steps,
        )

        return self.prepare_output(
            latents=latents,
            latent_num_frames=latent_num_frames,
            latent_height=latent_height,
            latent_width=latent_width,
            offload=offload,
            return_latents=return_latents,
            generator=generator,
            decode_timestep=decode_timestep,
            decode_noise_scale=decode_noise_scale,
            postprocessor_kwargs=postprocessor_kwargs,
        )
