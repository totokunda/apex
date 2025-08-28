import torch
from typing import Dict, Any, Callable, List, Union, Optional
from PIL import Image
import numpy as np
from .base import WanBaseEngine


class WanRecamEngine(WanBaseEngine):
    """WAN Recam Engine Implementation"""

    def run(
        self,
        camera_extrinsics: str | np.ndarray | torch.Tensor,
        source_video: Union[
            List[Image.Image], List[str], str, np.ndarray, torch.Tensor
        ],
        prompt: List[str] | str,
        negative_prompt: List[str] | str = None,
        cam_type: int = None,
        video: Union[
            List[Image.Image], List[str], str, np.ndarray, torch.Tensor
        ] = None,
        height: int = 480,
        width: int = 832,
        duration: int | str = 81,
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
        
        batch_size = prompt_embeds.shape[0]

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

        loaded_video = self._load_video(source_video, fps=fps)

        num_frames = self._parse_num_frames(duration, fps)
        if num_frames > len(loaded_video):
            num_frames = (
                len(loaded_video) // self.vae_scale_factor_temporal
            ) * self.vae_scale_factor_temporal + 1

        if isinstance(camera_extrinsics, str):
            camera_extrinsics = self.helpers["wan.recam"](
                camera_extrinsics, num_frames=num_frames, cam_type=cam_type
            ).to(self.device)
        elif isinstance(camera_extrinsics, np.ndarray):
            camera_extrinsics = torch.from_numpy(camera_extrinsics).to(self.device)
        else:
            camera_extrinsics = camera_extrinsics.to(self.device)

        preprocessed_video = self.video_processor.preprocess_video(
            loaded_video, height=height, width=width
        ).to(self.device, dtype=torch.float32)

        source_latents = self.vae_encode(
            preprocessed_video, offload=offload, sample_mode="mode"
        )

        if not self.transformer:
            self.load_component_by_type("transformer")

        transformer_dtype = self.component_dtypes["transformer"]

        self.to_device(self.transformer)

        prompt_embeds = prompt_embeds.to(self.device, dtype=transformer_dtype)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(
                self.device, dtype=transformer_dtype
            )

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

        latent_timestep = timesteps[:1].repeat(num_videos)

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
            num_frames,
            num_channels_latents=getattr(vae_config, "z_dim", 16),
            vae_scale_factor_spatial=vae_scale_factor_spatial,
            vae_scale_factor_temporal=vae_scale_factor_temporal,
            fps=fps,
            batch_size=batch_size,
            seed=seed,
            dtype=torch.float32,
            generator=generator,
        )

        if video is not None:
            video = self._load_video(video, fps=fps)
            preprocessed_video = self.video_processor.preprocess_video(
                video, height=height, width=width
            ).to(self.device, dtype=torch.float32)
            cond_latent = self.vae_encode(preprocessed_video, offload=offload)
            cond_latent = cond_latent[:, :, : latents.shape[2], :, :]
        else:
            cond_latent = None

        if cond_latent is not None:
            if hasattr(self.scheduler, "add_noise"):
                latents = self.scheduler.add_noise(
                    cond_latent, latents, latent_timestep
                )
            else:
                latents = self.scheduler.scale_noise(
                    latents, latent_timestep, cond_latent
                )

        if boundary_ratio is not None:
            boundary_timestep = boundary_ratio * getattr(
                self.scheduler.config, "num_train_timesteps", 1000
            )
        else:
            boundary_timestep = None

        source_latents = source_latents[:, :, : latents.shape[2], :, :]

        latents = self.denoise(
            boundary_timestep=boundary_timestep,
            timesteps=timesteps,
            latents=latents,
            source_latents=source_latents,
            transformer_kwargs=dict(
                encoder_hidden_states=prompt_embeds,
                attention_kwargs=attention_kwargs,
                cam_hidden_states=camera_extrinsics.to(transformer_dtype),
            ),
            unconditional_transformer_kwargs=(
                dict(
                    encoder_hidden_states=negative_prompt_embeds,
                    cam_hidden_states=camera_extrinsics.to(transformer_dtype),
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
            postprocessed_video = self._tensor_to_frames(video)
            return postprocessed_video
