import torch
from typing import Dict, Any, Callable, List, Union, Optional
from PIL import Image
import numpy as np
import torch.nn.functional as F
from .base import WanBaseEngine


class WanInpEngine(WanBaseEngine):
    """WAN Inpainting Engine Implementation for video inpainting with masks"""

    def run(
        self,
        reference_image: Union[
            Image.Image,
            List[Image.Image],
            List[str],
            str,
            np.ndarray,
            torch.Tensor,
            None,
        ] = None,
        video: Union[
            List[Image.Image], List[str], str, np.ndarray, torch.Tensor, None
        ] = None,
        mask: Union[
            Image.Image,
            List[Image.Image],
            List[str],
            str,
            np.ndarray,
            torch.Tensor,
            None,
        ] = None,
        process_first_mask_frame_only: bool = False,
        prompt: List[str] | str = None,
        negative_prompt: List[str] | str = None,
        duration: int | str = 16,
        height: int = 480,
        width: int = 832,
        num_inference_steps: int = 30,
        num_videos: int = 1,
        seed: int | None = None,
        fps: int = 16,
        guidance_scale: float = 5.0,
        use_cfg_guidance: bool = True,
        return_latents: bool = False,
        text_encoder_kwargs: Dict[str, Any] = {},
        attention_kwargs: Dict[str, Any] = {},
        render_on_step_callback: Callable = None,
        generator: torch.Generator | None = None,
        offload: bool = True,
        render_on_step: bool = False,
        timesteps: List[int] | None = None,
        timesteps_as_indices: bool = True,
        boundary_ratio: float | None = None,
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

        if not self.preprocessors or "clip" not in self.preprocessors:
            self.load_preprocessor_by_type("clip")

        self.to_device(self.preprocessors["clip"])

        transformer_config = self.load_config_by_type("transformer")

        transformer_dtype = self.component_dtypes["transformer"]

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

        if video is not None and mask is not None:
            pt, ph, pw = transformer_config.patch_size
            loaded_video = self._load_video(video)
            video_height, video_width = self.video_processor.get_default_height_width(
                loaded_video[0]
            )
            base = self.vae_scale_factor_spatial * ph
            if video_height * video_width > height * width:
                scale = min(width / video_width, height / video_height)
                video_height, video_width = int(video_height * scale), int(
                    video_width * scale
                )

            if video_height % base != 0 or video_width % base != 0:
                video_height = (video_height // base) * base
                video_width = (video_width // base) * base

            assert video_height * video_width <= height * width

            preprocessed_video = self.video_processor.preprocess_video(
                loaded_video, video_height, video_width
            )

            batch_size, latent_num_frames, _, _, _ = latents.shape
            loaded_mask = self._load_video(mask)
            preprocessed_mask = self.video_processor.preprocess_video(
                loaded_mask, video_height, video_width
            )
            preprocessed_mask = torch.clamp((preprocessed_mask + 1) / 2, min=0, max=1)

            if (preprocessed_mask == 0).all():
                mask_latents = torch.tile(
                    torch.zeros_like(latents)[:, :1].to(self.device, transformer_dtype),
                    [1, 4, 1, 1, 1],
                )
                masked_video_latents = torch.zeros_like(latents).to(
                    self.device, transformer_dtype
                )
            else:
                masked_video = preprocessed_video * (
                    torch.tile(preprocessed_mask, [1, 3, 1, 1, 1]) < 0.5
                )
                masked_video_latents = self._prepare_fun_control_latents(
                    masked_video, dtype=torch.float32, generator=generator
                )
                mask_condition = torch.concat(
                    [
                        torch.repeat_interleave(
                            preprocessed_mask[:, :, 0:1], repeats=4, dim=2
                        ),
                        preprocessed_mask[:, :, 1:],
                    ],
                    dim=2,
                )
                mask_condition = mask_condition.view(
                    batch_size, mask_condition.shape[2] // 4, 4, height, width
                )
                mask_condition = mask_condition.transpose(1, 2)
                latent_size = latents.size()
                batch_size, channels, num_frames, height, width = (
                    masked_video_latents.shape
                )
                inverse_mask_condition = 1 - mask_condition

                if process_first_mask_frame_only:
                    target_size = list(latent_size[2:])
                    target_size[0] = 1
                    first_frame_resized = F.interpolate(
                        inverse_mask_condition[:, :, 0:1, :, :],
                        size=target_size,
                        mode="trilinear",
                        align_corners=False,
                    )

                    target_size = list(latent_size[2:])
                    target_size[0] = target_size[0] - 1
                    if target_size[0] != 0:
                        remaining_frames_resized = F.interpolate(
                            inverse_mask_condition[:, :, 1:, :, :],
                            size=target_size,
                            mode="trilinear",
                            align_corners=False,
                        )
                        resized_mask = torch.cat(
                            [first_frame_resized, remaining_frames_resized], dim=2
                        )
                    else:
                        resized_mask = first_frame_resized
                else:
                    target_size = list(latent_size[2:])
                    resized_mask = F.interpolate(
                        inverse_mask_condition,
                        size=target_size,
                        mode="trilinear",
                        align_corners=False,
                    )
                mask_latents = resized_mask

            control_latents = torch.concat([mask_latents, masked_video_latents], dim=1)
        else:
            control_latents = torch.zeros_like(latents)

        if reference_image is not None and transformer_config.get(
            "add_ref_control", False
        ):
            loaded_image = self._load_image(reference_image)
            loaded_image, height, width = self._aspect_ratio_resize(
                loaded_image, max_area=height * width
            )
            preprocessed_image = (
                self.video_processor.preprocess(
                    loaded_image, height=height, width=width
                )
                .to(self.device, dtype=torch.float32)
                .unsqueeze(2)
            )

            reference_image_latents = self._prepare_fun_control_latents(
                preprocessed_image, dtype=torch.float32, generator=generator
            )
        else:
            reference_image_latents = torch.zeros_like(latents)[:, :, :1]

        if reference_image is not None:
            clip_image = reference_image
        else:
            clip_image = None

        if clip_image is not None:
            loaded_image = self._load_image(clip_image)
            loaded_image, height, width = self._aspect_ratio_resize(
                loaded_image, max_area=height * width
            )
            image_embeds = self.preprocessors["clip"](
                loaded_image, hidden_states_layer=-2
            ).to(self.device, dtype=transformer_dtype)
        else:
            image_embeds = None

        prompt_embeds = prompt_embeds.to(self.device, dtype=transformer_dtype)

        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(
                self.device, dtype=transformer_dtype
            )

        if offload:
            self._offload(self.preprocessors["clip"])

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

        if boundary_ratio is not None:
            boundary_timestep = boundary_ratio * getattr(
                self.scheduler.config, "num_train_timesteps", 1000
            )
        else:
            boundary_timestep = None

        latents = self.denoise(
            boundary_timestep=boundary_timestep,
            timesteps=timesteps,
            latents=latents,
            latent_condition=control_latents,
            transformer_kwargs=dict(
                encoder_hidden_states=prompt_embeds,
                encoder_hidden_states_image=image_embeds,
                encoder_hidden_states_full_ref=(
                    reference_image_latents.to(transformer_dtype)
                    if reference_image_latents is not None
                    else None
                ),
                attention_kwargs=attention_kwargs,
            ),
            unconditional_transformer_kwargs=(
                dict(
                    encoder_hidden_states=negative_prompt_embeds,
                    encoder_hidden_states_image=image_embeds,
                    encoder_hidden_states_full_ref=(
                        reference_image_latents.to(transformer_dtype)
                        if reference_image_latents is not None
                        else None
                    ),
                    attention_kwargs=attention_kwargs,
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

        if offload:
            self._offload(self.transformer)

        if return_latents:
            return latents
        else:
            video = self.vae_decode(latents, offload=offload)
            postprocessed_video = self._postprocess(video)
            return postprocessed_video
