import torch
from PIL import Image
import numpy as np
import torchvision.transforms.functional as F
import torch.nn.functional as F
import math
from torchvision import transforms
from typing import TYPE_CHECKING, Callable
from src.engine.base_engine import BaseEngine
from diffusers.video_processor import VideoProcessor
from diffusers.image_processor import VaeImageProcessor
from src.utils.progress import safe_emit_progress
from src.utils.cache import empty_cache
from .mlx import WanMLXDenoise

class WanShared(BaseEngine, WanMLXDenoise):
    """Base class for WAN engine implementations containing common functionality"""

    def __init__(self, yaml_path: str, **kwargs):
        super().__init__(yaml_path, **kwargs)
        self.vae_scale_factor_temporal = (
            2 ** sum(self.vae.temperal_downsample)
            if getattr(self.vae, "temperal_downsample", None)
            else 4
        )

        self.vae_scale_factor_spatial = (
            2 ** len(self.vae.temperal_downsample)
            if getattr(self.vae, "temperal_downsample", None)
            else 8
        )

        self.num_channels_latents = getattr(self.vae, "config", {}).get("z_dim", 16)

        self.video_processor = VideoProcessor(
            vae_scale_factor=kwargs.get(
                "vae_scale_factor", self.vae_scale_factor_spatial
            )
        )
        
        self.mask_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor_spatial, do_normalize=False, do_binarize=True, do_convert_grayscale=True
        )

   

    def _prepare_fun_control_latents(
        self, control, dtype=torch.float32, generator: torch.Generator | None = None
    ):
        """Prepare control latents for FUN implementation"""
        # resize the control to latents shape as we concatenate the control to the latents
        # we do that before converting to dtype to avoid breaking in case we're using cpu_offload
        # and half precision

        bs = 1
        new_control = []
        for i in range(0, control.shape[0], bs):
            control_bs = control[i : i + bs]
            control_bs = self.vae_encode(
                control_bs, sample_generator=generator, normalize_latents_dtype=dtype
            )
            new_control.append(control_bs)
        control = torch.cat(new_control, dim=0)

        return control
    

    def resize_and_centercrop(self, cond_image, target_size):
        """
        Resize image or tensor to the target size without padding.
        """

        # Get the original size
        if isinstance(cond_image, torch.Tensor):
            _, orig_h, orig_w = cond_image.shape
        else:
            orig_h, orig_w = cond_image.height, cond_image.width

        target_h, target_w = target_size

        # Calculate the scaling factor for resizing
        scale_h = target_h / orig_h
        scale_w = target_w / orig_w

        # Compute the final size
        scale = max(scale_h, scale_w)
        final_h = math.ceil(scale * orig_h)
        final_w = math.ceil(scale * orig_w)

        # Resize
        if isinstance(cond_image, torch.Tensor):
            if len(cond_image.shape) == 3:
                cond_image = cond_image[None]
            resized_tensor = F.interpolate(
                cond_image, size=(final_h, final_w), mode="nearest"
            ).contiguous()
            # crop
            cropped_tensor = transforms.functional.center_crop(
                resized_tensor, target_size
            )
            cropped_tensor = cropped_tensor.squeeze(0)
        else:
            resized_image = cond_image.resize(
                (final_w, final_h), resample=Image.BILINEAR
            )
            resized_image = np.array(resized_image)
            # tensor and crop
            resized_tensor = (
                torch.from_numpy(resized_image)[None, ...]
                .permute(0, 3, 1, 2)
                .contiguous()
            )
            cropped_tensor = transforms.functional.center_crop(
                resized_tensor, target_size
            )
            cropped_tensor = cropped_tensor[:, :, None, :, :]

        return cropped_tensor
    
    def _resize_mask(self, mask, latent, process_first_frame_only=True):
        latent_size = latent.size()
        batch_size, channels, num_frames, height, width = mask.shape

        if process_first_frame_only:
            target_size = list(latent_size[2:])
            target_size[0] = 1
            first_frame_resized = F.interpolate(
                mask[:, :, 0:1, :, :],
                size=target_size,
                mode='trilinear',
                align_corners=False
            )

            target_size = list(latent_size[2:])
            target_size[0] = target_size[0] - 1
            if target_size[0] != 0:
                remaining_frames_resized = F.interpolate(
                    mask[:, :, 1:, :, :],
                    size=target_size,
                    mode='trilinear',
                    align_corners=False
                )
                resized_mask = torch.cat([first_frame_resized, remaining_frames_resized], dim=2)
            else:
                resized_mask = first_frame_resized
        else:
            target_size = list(latent_size[2:])
            resized_mask = F.interpolate(
                mask,
                size=target_size,
                mode='trilinear',
                align_corners=False
            )
        return resized_mask

    def _render_step(self, latents: torch.Tensor, render_on_step_callback: Callable):
        self.logger.info(f"Rendering step for model type: {self.model_type}")
        if self.model_type == 't2i':
            tensor_image = self.vae_decode(latents)[:, :, 0]
            image = self._tensor_to_frame(tensor_image)
            render_on_step_callback(image[0])
        else:
            super()._render_step(latents, render_on_step_callback)
            
    
    def _encode_ip_image(
        self,
        ip_image: Image.Image | str | np.ndarray | torch.Tensor,
        dtype: torch.dtype = None,
    ):
        ip_image = self._load_image(ip_image)
        ip_image = (
            torch.tensor(np.array(ip_image)).permute(2, 0, 1).float() / 255.0
        )  # [3, H, W]
        ip_image = ip_image.unsqueeze(1).unsqueeze(0).to(dtype=dtype)  # [B, 3, 1, H, W]
        ip_image = ip_image * 2 - 1

        encoded_image = self.vae_encode(ip_image, sample_mode="mode", dtype=dtype)
        return encoded_image
            

    def moe_denoise(self, *args, **kwargs) -> torch.Tensor:
        timesteps = kwargs.get("timesteps", None)
        latents = kwargs.get("latents", None)
        latent_condition = kwargs.get("latent_condition", None)
        transformer_dtype = kwargs.get("transformer_dtype", None)
        use_cfg_guidance = kwargs.get("use_cfg_guidance", True)
        render_on_step = kwargs.get("render_on_step", False)
        render_on_step_callback = kwargs.get("render_on_step_callback", None)
        denoise_progress_callback = kwargs.get("denoise_progress_callback", None)
        scheduler = kwargs.get("scheduler", None)
        guidance_scale = kwargs.get("guidance_scale", 5.0)
        boundary_timestep = kwargs.get("boundary_timestep", None)
        transformer_kwargs = kwargs.get("transformer_kwargs", {})
        unconditional_transformer_kwargs = kwargs.get("unconditional_transformer_kwargs", {})
        transformer_kwargs.pop("encoder_hidden_states_image", None)
        unconditional_transformer_kwargs.pop("encoder_hidden_states_image", None)
        mask_kwargs = kwargs.get("mask_kwargs", {})
        mask = mask_kwargs.get("mask", None)
        masked_video_latents = mask_kwargs.get("masked_video_latents", None)
        render_on_step_interval = kwargs.get("render_on_step_interval", 3)

        total_steps = len(timesteps) if timesteps is not None else 0
        safe_emit_progress(denoise_progress_callback, 0.0, "Starting denoise")

        with self._progress_bar(len(timesteps), desc=f"Sampling MOE") as pbar:
            total_steps = len(timesteps)
            for i, t in enumerate(timesteps):

                if latent_condition is not None:
                    latent_model_input = torch.cat(
                        [latents, latent_condition], dim=1
                    ).to(transformer_dtype)
                else:
                    latent_model_input = latents.to(transformer_dtype)

                timestep = t.expand(latents.shape[0])

                if boundary_timestep is not None and t >= boundary_timestep:
                   
                    if hasattr(self, "transformer_2") and self.transformer_2:
                        safe_emit_progress(
                            denoise_progress_callback,
                            float(i) / float(total_steps) if total_steps else 0.0,
                            "Offloading previous transformer",
                        )
                        self._offload(self.transformer)
                        setattr(self, "transformer", None)
                        empty_cache()

                    if not self.transformer:
                        safe_emit_progress(
                            denoise_progress_callback,
                            float(i) / float(total_steps) if total_steps else 0.0,
                            "Loading new transformer",
                        )

                        self.load_component_by_name("transformer")
                        self.to_device(self.transformer)
                        safe_emit_progress(
                            denoise_progress_callback,
                            float(i) / float(total_steps) if total_steps else 0.0,
                            "New transformer ready",
                        )

                    transformer = self.transformer

                    if isinstance(guidance_scale, list):
                        guidance_scale = guidance_scale[0]
                else:
                    if self.transformer:
                        safe_emit_progress(
                            denoise_progress_callback,
                            float(i) / float(total_steps) if total_steps else 0.0,
                            "Switching model boundary, offloading previous transformer",
                        )
                        self._offload(self.transformer)
                        setattr(self, "transformer", None)
                        empty_cache()

                    if not hasattr(self, "transformer_2") or not self.transformer_2:
                        safe_emit_progress(
                            denoise_progress_callback,
                            float(i) / float(total_steps) if total_steps else 0.0,
                            "Loading alternate transformer",
                        )
                        self.load_component_by_name("transformer_2")
                        self.to_device(self.transformer_2)
                        safe_emit_progress(
                            denoise_progress_callback,
                            float(i) / float(total_steps) if total_steps else 0.0,
                            "Alternate transformer ready",
                        )

                    transformer = self.transformer_2
                    if isinstance(guidance_scale, list):
                        guidance_scale = guidance_scale[1]
                    # Standard denoising
                noise_pred = transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    return_dict=False,
                    **kwargs.get("transformer_kwargs", {}),
                )[0]
                if use_cfg_guidance and kwargs.get(
                    "unconditional_transformer_kwargs", None
                ):
                    uncond_noise_pred = transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep,
                        return_dict=False,
                        **kwargs.get("unconditional_transformer_kwargs", {}),
                    )[0]
                    noise_pred = uncond_noise_pred + guidance_scale * (
                        noise_pred - uncond_noise_pred
                    )
                latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]
            
                if self.vae_scale_factor_spatial >= 16 and mask is not None and not mask[:, :, 0, :, :].any():
                    latents = (1 - mask) * masked_video_latents + mask * latents

                if render_on_step and render_on_step_callback and ((i + 1) % render_on_step_interval == 0 or i == 0) and i != len(timesteps) - 1:
                    self._render_step(latents, render_on_step_callback)
                pbar.update(1)
                safe_emit_progress(
                    denoise_progress_callback,
                    float(i + 1) / float(total_steps),
                    f"Denoising step {i + 1}/{total_steps}",
                )

            self.logger.info("Denoising completed.")

        return latents

    def base_denoise(self, *args, **kwargs) -> torch.Tensor:
        timesteps = kwargs.get("timesteps", None)
        latents = kwargs.get("latents", None)
        latent_condition = kwargs.get("latent_condition", None)
        transformer_dtype = kwargs.get("transformer_dtype", None)
        use_cfg_guidance = kwargs.get("use_cfg_guidance", True)
        render_on_step = kwargs.get("render_on_step", False)
        render_on_step_callback = kwargs.get("render_on_step_callback", None)
        denoise_progress_callback = kwargs.get("denoise_progress_callback", None)
        scheduler = kwargs.get("scheduler", None)
        guidance_scale = kwargs.get("guidance_scale", 5.0)
        expand_timesteps = kwargs.get("expand_timesteps", False)
        first_frame_mask = kwargs.get("first_frame_mask", None)
        ip_image = kwargs.get("ip_image", None)
        render_on_step_interval = kwargs.get("render_on_step_interval", 3)

        total_steps = len(timesteps) if timesteps is not None else 0
        safe_emit_progress(denoise_progress_callback, 0.0, "Starting denoise")

        if ip_image is not None:
            ip_image_latent = self._encode_ip_image(ip_image, dtype=transformer_dtype)
        else:
            ip_image_latent = None

        if expand_timesteps and first_frame_mask is not None:
            mask = torch.ones(latents.shape, dtype=torch.float32, device=self.device)
        else:
            mask = None

        if not self.transformer:
            self.load_component_by_type("transformer")
        self.to_device(self.transformer)

        model_type_str = getattr(self, "model_type", "WAN")

        with self._progress_bar(
            len(timesteps), desc=f"Sampling {model_type_str}"
        ) as pbar:
            total_steps = len(timesteps)
            for i, t in enumerate(timesteps):
                if expand_timesteps:
                    # seq_len: num_latent_frames * latent_height//2 * latent_width//2
                    if latent_condition is not None and first_frame_mask is not None:
                        latent_model_input = (
                            1 - first_frame_mask
                        ) * latent_condition + first_frame_mask * latents
                        latent_model_input = latent_model_input.to(transformer_dtype)
                        temp_ts = (first_frame_mask[0][0][:, ::2, ::2] * t).flatten()
                    else:
                        latent_model_input = latents.to(transformer_dtype)
                        temp_ts = (mask[0][0][:, ::2, ::2] * t).flatten()

                    # batch_size, seq_len
                    timestep = temp_ts.unsqueeze(0).expand(latents.shape[0], -1)
                else:
                    timestep = t.expand(latents.shape[0])

                    if latent_condition is not None:
                        latent_model_input = torch.cat(
                            [latents, latent_condition], dim=1
                        ).to(transformer_dtype)
                    else:
                        latent_model_input = latents.to(transformer_dtype)

                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    ip_image_hidden_states=ip_image_latent,
                    timestep=timestep,
                    return_dict=False,
                    **kwargs.get("transformer_kwargs", {}),
                )[0]

                ip_image_latent = None

                if use_cfg_guidance and kwargs.get(
                    "unconditional_transformer_kwargs", None
                ):
                    uncond_noise_pred = self.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep,
                        return_dict=False,
                        **kwargs.get("unconditional_transformer_kwargs", {}),
                    )[0]
                    noise_pred = uncond_noise_pred + guidance_scale * (
                        noise_pred - uncond_noise_pred
                    )

                latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if render_on_step and render_on_step_callback and ((i + 1) % render_on_step_interval == 0 or i == 0) and i != len(timesteps) - 1:
                    self._render_step(latents, render_on_step_callback)
                pbar.update(1)
                safe_emit_progress(
                    denoise_progress_callback,
                    float(i + 1) / float(total_steps),
                    f"Denoising step {i + 1}/{total_steps}",
                )

            if expand_timesteps and first_frame_mask is not None:
                latents = (
                    1 - first_frame_mask
                ) * latent_condition + first_frame_mask * latents

            self.logger.info("Denoising completed.")

        return latents