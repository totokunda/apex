import torch
from typing import Dict, Any, Callable, List, Union, Optional
from PIL import Image
import numpy as np

from .base import WanBaseEngine


class WanI2VEngine(WanBaseEngine):
    """WAN Image-to-Video Engine Implementation"""

    def run(
        self,
        image: Union[
            Image.Image, List[Image.Image], List[str], str, np.ndarray, torch.Tensor
        ],
        prompt: List[str] | str,
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
        expand_timesteps: bool = False,
        ip_image: Image.Image | str | np.ndarray | torch.Tensor = None,
        enhance_kwargs: Dict[str, Any] = {},
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

        loaded_image = self._load_image(image)

        loaded_image, height, width = self._aspect_ratio_resize(
            loaded_image,
            max_area=height * width,
            mod_value=32 if expand_timesteps else 16,
        )

        preprocessed_image = self.video_processor.preprocess(
            loaded_image, height=height, width=width
        ).to(self.device, dtype=torch.float32)

        if not self.transformer:
            self.load_component_by_type("transformer")

        transformer_dtype = self.component_dtypes["transformer"]

        self.to_device(self.transformer)

        if boundary_ratio is None and not expand_timesteps:
            image_embeds = self.helpers["clip"](
                loaded_image, hidden_states_layer=-2
            ).to(self.device, dtype=transformer_dtype)
        else:
            image_embeds = None

        prompt_embeds = prompt_embeds.to(self.device, dtype=transformer_dtype)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(
                self.device, dtype=transformer_dtype
            )

        if offload and boundary_ratio is None and not expand_timesteps:
            self._offload(self.helpers["clip"])

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

        vae_config = self.load_config_by_type("vae")
        vae_scale_factor_spatial = getattr(
            vae_config, "scale_factor_spatial", self.vae_scale_factor_spatial
        )
        vae_scale_factor_temporal = getattr(
            vae_config, "scale_factor_temporal", self.vae_scale_factor_temporal
        )

        latents = self._get_latents(
            height,
            width,
            duration,
            num_channels_latents=getattr(vae_config, "z_dim", 16),
            vae_scale_factor_spatial=vae_scale_factor_spatial,
            vae_scale_factor_temporal=vae_scale_factor_temporal,
            fps=fps,
            num_videos=num_videos,
            seed=seed,
            dtype=torch.float32,
            generator=generator,
        )

        if preprocessed_image.ndim == 4:
            preprocessed_image = preprocessed_image.unsqueeze(2)

        video_condition = torch.cat(
            [
                preprocessed_image,
                preprocessed_image.new_zeros(
                    preprocessed_image.shape[0],
                    preprocessed_image.shape[1],
                    num_frames - 1,
                    height,
                    width,
                ),
            ],
            dim=2,
        )

        latent_condition = self.vae_encode(
            video_condition,
            offload=offload,
            dtype=latents.dtype,
            normalize_latents_dtype=latents.dtype,
        )

        batch_size, _, num_latent_frames, latent_height, latent_width = latents.shape

        if expand_timesteps:
            first_frame_mask = torch.ones(
                1,
                1,
                num_latent_frames,
                latent_height,
                latent_width,
                dtype=latents.dtype,
                device=latents.device,
            )
            first_frame_mask[:, :, 0] = 0
        else:
            mask_lat_size = torch.ones(
                batch_size,
                1,
                num_frames,
                latent_height,
                latent_width,
                device=self.device,
            )

            mask_lat_size[:, :, list(range(1, num_frames))] = 0
            first_frame_mask = mask_lat_size[:, :, 0:1]
            first_frame_mask = torch.repeat_interleave(
                first_frame_mask, dim=2, repeats=self.vae_scale_factor_temporal
            )

            mask_lat_size = torch.concat(
                [first_frame_mask, mask_lat_size[:, :, 1:, :]], dim=2
            )
            mask_lat_size = mask_lat_size.view(
                batch_size,
                -1,
                self.vae_scale_factor_temporal,
                latent_height,
                latent_width,
            )

            mask_lat_size = mask_lat_size.transpose(1, 2)
            mask_lat_size = mask_lat_size.to(latents.device)

            latent_condition = torch.concat([mask_lat_size, latent_condition], dim=1)

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
            latent_condition=latent_condition,
            transformer_kwargs=dict(
                encoder_hidden_states=prompt_embeds,
                encoder_hidden_states_image=image_embeds,
                attention_kwargs=attention_kwargs,
                enhance_kwargs=enhance_kwargs,
            ),
            unconditional_transformer_kwargs=(
                dict(
                    encoder_hidden_states=negative_prompt_embeds,
                    encoder_hidden_states_image=image_embeds,
                    attention_kwargs=attention_kwargs,
                    enhance_kwargs=enhance_kwargs,
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
            expand_timesteps=expand_timesteps,
            first_frame_mask=first_frame_mask,
            ip_image=ip_image,
        )

        if offload:
            self._offload(self.transformer)

        if return_latents:
            return latents
        else:
            video = self.vae_decode(latents, offload=offload)
            postprocessed_video = self._tensor_to_frames(video)
            return postprocessed_video
