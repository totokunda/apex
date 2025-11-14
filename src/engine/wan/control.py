import torch
from typing import Dict, Any, Callable, List, Union, Optional
from PIL import Image
import numpy as np
import torch.nn.functional as F
from src.helpers.wan.fun_camera import Camera
from .shared import WanShared
from einops import rearrange

class WanControlEngine(WanShared):
    """WAN Control Engine Implementation for camera control and video guidance"""

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
        start_image: Union[
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
        control_video: Union[
            List[Image.Image], List[str], str, np.ndarray, torch.Tensor, None
        ] = None,
        mask: Union[
            List[Image.Image], List[str], str, np.ndarray, torch.Tensor, None
        ] = None,
        camera_poses: Union[List[float], str, List[Camera], Camera, None] = None,
        process_first_mask_frame_only: bool = True, 
        prompt: List[str] | str = None,
        negative_prompt: List[str] | str = None,
        duration: int | str = 16,
        height: int = 480,
        width: int = 832,
        num_inference_steps: int = 30,
        num_videos: int = 1,
        seed: int | None = None,
        fps: int = 81,
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
        boundary_ratio: float | None = 0.875,
        enhance_kwargs: Dict[str, Any] = {},
        scheduler_kwargs: Dict[str, Any] = {},
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
        
        num_frames = self._parse_num_frames(duration, fps)
        
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

        if start_image is not None:
            loaded_image = self._load_image(start_image)

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

            start_image_latents = self._prepare_fun_control_latents(
                preprocessed_image, dtype=torch.float32, generator=generator
            )

        transformer_config = self.load_config_by_type("transformer")
        
        transformer_dtype = self.component_dtypes["transformer"]
        
        latents = self._get_latents(
            height,
            width,
            duration,
            fps=fps,
            batch_size=batch_size,
            seed=seed,
            dtype=torch.float32,
            generator=generator,
        )
        
        if self.denoise_type == "moe" and start_image is not None:
            # convert start image into video 
            start_image = self._load_image(start_image)
            video = [start_image] * num_frames
            mask = [Image.new("RGB", (width, height), (0, 0, 0))] * num_frames
            mask[0] = Image.new("RGB", (width, height), (255, 255, 255))
            start_image_latents_in = None
        elif start_image is not None:
            start_image_latents_in = torch.zeros_like(latents)
            if start_image_latents_in.shape[2] > 1:
                start_image_latents_in[:, :, :1] = start_image_latents
        else:
            start_image_latents_in = torch.zeros_like(latents)

        if camera_poses is not None:
            control_latents = None
            if isinstance(camera_poses, Camera):
                camera_poses = [camera_poses]
            camera_preprocessor = self.helpers["wan.fun_camera"]
            control_camera_video = camera_preprocessor(
                camera_poses, H=height, W=width, device=self.device
            )
            control_camera_latents = torch.concat(
                [
                    torch.repeat_interleave(
                        control_camera_video[:, :, 0:1], repeats=4, dim=2
                    ),
                    control_camera_video[:, :, 1:],
                ],
                dim=2,
            ).transpose(1, 2)

            # Reshape, transpose, and view into desired shape
            b, f, c, h, w = control_camera_latents.shape
            control_camera_latents = (
                control_camera_latents.contiguous()
                .view(b, f // 4, 4, c, h, w)
                .transpose(2, 3)
            )
            control_camera_latents = (
                control_camera_latents.contiguous()
                .view(b, f // 4, c * 4, h, w)
                .transpose(1, 2)
            )

        elif control_video is not None:
            pt, ph, pw = transformer_config.get("patch_size", (1, 2, 2))
            loaded_video = self._load_video(control_video, fps=fps)
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
            
            control_video = torch.from_numpy(np.array([np.array(frame) for frame in loaded_video]))[:num_frames]
            control_video = control_video.permute([3, 0, 1, 2]).unsqueeze(0) / 255
            
            video_length = control_video.shape[2]
            control_video = self.video_processor.preprocess(rearrange(control_video, "b c f h w -> (b f) c h w"), height=height, width=width) 
            control_video = control_video.to(dtype=torch.float32)
            control_video = rearrange(control_video, "(b f) c h w -> b c f h w", f=video_length)

            control_latents = self._prepare_fun_control_latents(
                control_video, dtype=torch.float32, generator=generator
            )
            control_camera_latents = None
        else:
            control_latents = torch.zeros_like(latents)
            control_camera_latents = None
        
        
        
        if video is not None and self.denoise_type == "moe":
            if mask is not None:
                mask = self._load_video(mask, fps=fps)
                mask = torch.from_numpy(np.array([np.array(frame) for frame in mask]))[:num_frames]
                mask = mask.permute([3, 0, 1, 2]).unsqueeze(0) / 255
                mask = self.video_processor.preprocess(rearrange(mask, "b c f h w -> (b f) c h w"), height=height, width=width) 
                mask = mask.to(dtype=torch.float32)
                mask = rearrange(mask, "(b f) c h w -> b c f h w", f=num_frames)
                
            if mask is None or (mask == 1.0).all():
                mask_latents = torch.tile(
                    torch.zeros_like(latents)[:, :1].to(self.device, transformer_dtype), [1, 4, 1, 1, 1]
                )
                masked_video_latents = torch.zeros_like(latents).to(self.device, transformer_dtype)
                if self.vae_scale_factor_spatial >= 16:
                    _mask = torch.ones_like(latents).to(self.device, transformer_dtype)[:, :1].to(self.device, transformer_dtype)
                else:
                    _mask = None
            else:
                # Ensure we preprocess the provided input video, not an undefined variable
                loaded_video = self._load_video(video, fps=fps)
                video = torch.from_numpy(np.array([np.array(frame) for frame in loaded_video]))[:num_frames]
                video = video.permute([3, 0, 1, 2]).unsqueeze(0) / 255
                video = self.video_processor.preprocess(rearrange(video, "b c f h w -> (b f) c h w"), height=height, width=width) 
                video = video.to(dtype=torch.float32)
                video = rearrange(video, "(b f) c h w -> b c f h w", f=video_length)
                
                bs, _, video_length, height, width = video.size()
                mask_condition = self.mask_processor.preprocess(rearrange(mask, "b c f h w -> (b f) c h w"), height=height, width=width) 
                mask_condition = mask_condition.to(dtype=torch.float32)
                mask_condition = rearrange(mask_condition, "(b f) c h w -> b c f h w", f=video_length)

                masked_video = video * (torch.tile(mask_condition, [1, 3, 1, 1, 1]) < 0.5)
                masked_video_latents = self._prepare_fun_control_latents(
                    masked_video, dtype=torch.float32, generator=generator
                )
                
                mask_condition = torch.concat(
                    [
                        torch.repeat_interleave(mask_condition[:, :, 0:1], repeats=4, dim=2), 
                        mask_condition[:, :, 1:]
                    ], dim=2
                )
                mask_condition = mask_condition.view(bs, mask_condition.shape[2] // 4, 4, height, width)
                mask_condition = mask_condition.transpose(1, 2)
                mask_latents = self._resize_mask(1 - mask_condition, masked_video_latents, process_first_mask_frame_only).to(self.device, transformer_dtype) 

                if self.vae_scale_factor_spatial >= 16:
                    _mask = F.interpolate(mask_condition[:, :1], size=latents.size()[-3:], mode='trilinear', align_corners=True).to(self.device, transformer_dtype)
                    if not _mask[:, :, 0, :, :].any():
                        _mask[:, :, 1:, :, :] = 1
                        latents = (1 - _mask) * masked_video_latents + _mask * latents 
                else:
                    _mask = None
                
        elif self.denoise_type == "moe":
            mask_latents = torch.tile(
                    torch.zeros_like(latents)[:, :1].to(self.device, transformer_dtype), [1, 4, 1, 1, 1]
                )
            masked_video_latents = torch.zeros_like(latents).to(self.device, transformer_dtype)
            if self.vae_scale_factor_spatial >= 16:
                _mask = torch.ones_like(latents).to(self.device, transformer_dtype)[:, :1].to(self.device, transformer_dtype)
            else:
                _mask = None
        else:
            mask_latents = None
            masked_video_latents = None
            _mask = None

        if transformer_config.get("add_ref_conv", False):
            if reference_image is not None:
                loaded_image = self._load_image(reference_image)
                loaded_image = loaded_image.resize((width, height))
                preprocessed_image = (
                    self.video_processor.preprocess(
                        loaded_image
                    )
                    .to(self.device, dtype=torch.float32)
                    .unsqueeze(2)
                )

                reference_image_latents = self._prepare_fun_control_latents(
                    preprocessed_image, dtype=torch.float32, generator=generator
                )[:, :, 0]
            else:
                reference_image_latents = torch.zeros_like(latents)[:, :, 0]
        else:
            reference_image_latents = None
        
        if reference_image is not None:
            clip_image = reference_image
        elif start_image is not None:
            clip_image = start_image

        if clip_image is not None and self.denoise_type != "moe":
            loaded_image = self._load_image(clip_image)
            loaded_image, height, width = self._aspect_ratio_resize(
                loaded_image, max_area=height * width
            )
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

        if offload and image_embeds is not None:
            self._offload(self.helpers["clip"])

        if not self.scheduler:
            self.load_component_by_type("scheduler")
        self.to_device(self.scheduler)

        scheduler = self.scheduler
        scheduler.set_timesteps(
            num_inference_steps if timesteps is None else 1000, device=self.device
        )

        if boundary_ratio is not None:
            boundary_timestep = boundary_ratio * getattr(
                getattr(self.scheduler, "config", self.scheduler), "num_train_timesteps", 1000
            )
        else:
            boundary_timestep = None

        timesteps, num_inference_steps = self._get_timesteps(
            scheduler=scheduler,
            timesteps=timesteps,
            timesteps_as_indices=timesteps_as_indices,
            num_inference_steps=num_inference_steps,
            **scheduler_kwargs,
        )

        if control_latents is not None and start_image_latents_in is not None and mask_latents is None:
            control_latents = torch.concat(
                [
                    control_latents,
                    start_image_latents_in,
                ],
                dim=1,
            )
        elif control_latents is None and start_image_latents_in is not None:
            control_latents = start_image_latents_in
        
        if mask_latents is not None and masked_video_latents is not None:
            mask_latents = torch.concat(
                [
                    mask_latents,
                    masked_video_latents
                ],
                dim=1,
            )

        if mask_latents is not None and control_latents is not None:
            control_latents = torch.concat(
                [
                    control_latents,
                    mask_latents
                ],
                dim=1,
            )
        elif mask_latents is not None and control_latents is None:
            control_latents = mask_latents
            

        latents = self.denoise(
            boundary_timestep=boundary_timestep,
            timesteps=timesteps,
            latents=latents,
            latent_condition=control_latents,
            mask_kwargs=dict(
                mask=_mask,
                masked_video_latents=masked_video_latents,
            ),
            transformer_kwargs=dict(
                encoder_hidden_states=prompt_embeds,
                encoder_hidden_states_image=image_embeds,
                encoder_hidden_states_camera=(
                    control_camera_latents.to(transformer_dtype)
                    if control_camera_latents is not None
                    else None
                ),
                encoder_hidden_states_full_ref=(
                    reference_image_latents.to(transformer_dtype)
                    if reference_image_latents is not None
                    else None
                ),
                attention_kwargs=attention_kwargs,
                enhance_kwargs=enhance_kwargs,
            ),
            unconditional_transformer_kwargs=(
                dict(
                    encoder_hidden_states=negative_prompt_embeds,
                    encoder_hidden_states_image=image_embeds,
                    encoder_hidden_states_camera=(
                        control_camera_latents.to(transformer_dtype)
                        if control_camera_latents is not None
                        else None
                    ),
                    encoder_hidden_states_full_ref=(
                        reference_image_latents.to(transformer_dtype)
                        if reference_image_latents is not None
                        else None
                    ),
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
        )

        if offload:
            self._offload(self.transformer)

        if return_latents:
            return latents
        else:
            video = self.vae_decode(latents, offload=offload)
            postprocessed_video = self._tensor_to_frames(video)
            return postprocessed_video
