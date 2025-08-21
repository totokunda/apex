import torch
import math
from typing import Dict, Any, Callable, List, Union, Optional
from diffusers.utils.torch_utils import randn_tensor

class LTXBaseEngine:
    """Base class for LTX engine implementations"""

    def __init__(self, main_engine):
        self.main_engine = main_engine
        # Delegate properties to main engine
        for attr in [
            "device",
            "logger",
            "text_encoder",
            "transformer",
            "scheduler",
            "vae",
            "preprocessors",
            "component_dtypes",
        ]:
            setattr(self, attr, getattr(main_engine, attr, None))

    def __getattr__(self, name):
        """Delegate any missing attributes to main engine"""
        return getattr(self.main_engine, name)

    def _get_latents(
        self,
        height: int,
        width: int,
        duration: int | str,
        fps: int = 16,
        num_videos: int = 1,
        num_channels_latents: int = None,
        seed: int | None = None,
        dtype: torch.dtype = None,
        layout: torch.layout = None,
        generator: torch.Generator | None = None,
        return_generator: bool = False,
        parse_frames: bool = True,
    ):
        if parse_frames or isinstance(duration, str):
            num_frames = self._parse_num_frames(duration, fps)
            latent_num_frames = math.ceil(
                (num_frames + 3) / self.vae_scale_factor_temporal
            )
        else:
            latent_num_frames = duration

        latent_height = math.ceil(height / self.vae_scale_factor_spatial)
        latent_width = math.ceil(width / self.vae_scale_factor_spatial)

        if seed is not None and generator is not None:
            self.logger.warning(
                f"Both `seed` and `generator` are provided. `seed` will be ignored."
            )

        if generator is None:
            device = self.device
            if seed is not None:
                generator = torch.Generator(device=device).manual_seed(seed)
        else:
            device = generator.device

        noise = torch.randn(
            (
                num_videos,
                num_channels_latents or self.num_channels_latents,
                latent_num_frames,
                latent_height,
                latent_width,
            ),
            device=device,
            dtype=dtype,
            generator=generator,
            layout=layout or torch.strided,
        )

        if return_generator:
            return noise, generator
        else:
            return noise

    def trim_conditioning_sequence(
        self, start_frame: int, sequence_num_frames: int, target_num_frames: int
    ):
        """
        Trim a conditioning sequence to the allowed number of frames.

        Args:
            start_frame (int): The target frame number of the first frame in the sequence.
            sequence_num_frames (int): The number of frames in the sequence.
            target_num_frames (int): The target number of frames in the generated video.
        Returns:
            int: updated sequence length
        """
        scale_factor = self.vae_temporal_compression_ratio
        num_frames = min(sequence_num_frames, target_num_frames - start_frame)
        # Trim down to a multiple of temporal_scale_factor frames plus 1
        num_frames = (num_frames - 1) // scale_factor * scale_factor + 1
        return num_frames

    @staticmethod
    def add_noise_to_image_conditioning_latents(
        t: float,
        init_latents: torch.Tensor,
        latents: torch.Tensor,
        noise_scale: float,
        conditioning_mask: torch.Tensor,
        generator,
        eps=1e-6,
    ):
        """
        Add timestep-dependent noise to the hard-conditioning latents. This helps with motion continuity, especially
        when conditioned on a single frame.
        """
        noise = randn_tensor(
            latents.shape,
            generator=generator,
            device=latents.device,
            dtype=latents.dtype,
        )
        # Add noise only to hard-conditioning latents (conditioning_mask = 1.0)
        need_to_noise = (conditioning_mask > 1.0 - eps).unsqueeze(-1)
        noised_latents = init_latents + noise_scale * noise * (t**2)
        latents = torch.where(need_to_noise, noised_latents, latents)
        return latents

    @staticmethod
    def _pack_latents(
        latents: torch.Tensor, patch_size: int = 1, patch_size_t: int = 1
    ) -> torch.Tensor:
        # Unpacked latents of shape are [B, C, F, H, W] are patched into tokens of shape [B, C, F // p_t, p_t, H // p, p, W // p, p].
        # The patch dimensions are then permuted and collapsed into the channel dimension of shape:
        # [B, F // p_t * H // p * W // p, C * p_t * p * p] (an ndim=3 tensor).
        # dim=0 is the batch size, dim=1 is the effective video sequence length, dim=2 is the effective number of input features
        batch_size, num_channels, num_frames, height, width = latents.shape
        post_patch_num_frames = num_frames // patch_size_t
        post_patch_height = height // patch_size
        post_patch_width = width // patch_size
        latents = latents.reshape(
            batch_size,
            -1,
            post_patch_num_frames,
            patch_size_t,
            post_patch_height,
            patch_size,
            post_patch_width,
            patch_size,
        )
        latents = latents.permute(0, 2, 4, 6, 1, 3, 5, 7).flatten(4, 7).flatten(1, 3)
        return latents

    @staticmethod
    def _prepare_video_ids(
        batch_size: int,
        num_frames: int,
        height: int,
        width: int,
        patch_size: int = 1,
        patch_size_t: int = 1,
        device: torch.device = None,
    ) -> torch.Tensor:
        latent_sample_coords = torch.meshgrid(
            torch.arange(0, num_frames, patch_size_t, device=device),
            torch.arange(0, height, patch_size, device=device),
            torch.arange(0, width, patch_size, device=device),
            indexing="ij",
        )
        latent_sample_coords = torch.stack(latent_sample_coords, dim=0)
        latent_coords = latent_sample_coords.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1)
        latent_coords = latent_coords.reshape(
            batch_size, -1, num_frames * height * width
        )

        return latent_coords

    @staticmethod
    def _scale_video_ids(
        video_ids: torch.Tensor,
        scale_factor: int = 32,
        scale_factor_t: int = 8,
        frame_index: int = 0,
        device: torch.device = None,
    ) -> torch.Tensor:
        scaled_latent_coords = (
            video_ids
            * torch.tensor(
                [scale_factor_t, scale_factor, scale_factor], device=video_ids.device
            )[None, :, None]
        )
        scaled_latent_coords[:, 0] = (
            scaled_latent_coords[:, 0] + 1 - scale_factor_t
        ).clamp(min=0)
        scaled_latent_coords[:, 0] += frame_index

        return scaled_latent_coords

    def get_timesteps(self, sigmas, timesteps, num_inference_steps, strength):
        num_steps = min(int(num_inference_steps * strength), num_inference_steps)
        start_index = max(num_inference_steps - num_steps, 0)
        sigmas = sigmas[start_index:]
        timesteps = timesteps[start_index:]
        return sigmas, timesteps, num_inference_steps - start_index

    @staticmethod
    def _unpack_latents(
        latents: torch.Tensor,
        num_frames: int,
        height: int,
        width: int,
        patch_size: int = 1,
        patch_size_t: int = 1,
    ) -> torch.Tensor:
        # Packed latents of shape [B, S, D] (S is the effective video sequence length, D is the effective feature dimensions)
        # are unpacked and reshaped into a video tensor of shape [B, C, F, H, W]. This is the inverse operation of
        # what happens in the `_pack_latents` method.
        batch_size = latents.size(0)
        latents = latents.reshape(
            batch_size,
            num_frames,
            height,
            width,
            -1,
            patch_size_t,
            patch_size,
            patch_size,
        )
        latents = (
            latents.permute(0, 4, 1, 5, 2, 6, 3, 7)
            .flatten(6, 7)
            .flatten(4, 5)
            .flatten(2, 3)
        )
        return latents

    def prepare_output(
        self,
        latents: torch.Tensor,
        latent_num_frames: int,
        latent_height: int,
        latent_width: int,
        offload: bool = True,
        return_latents: bool = False,
        generator: torch.Generator | None = None,
        decode_timestep: Union[float, List[float]] = 0.0,
        decode_noise_scale: Optional[Union[float, List[float]]] = None,
        dtype: torch.dtype = None,
        upsample_latents: bool = False,
        upsample_kwargs: Dict[str, Any] = None,
    ):
        latents = self._unpack_latents(
            latents,
            latent_num_frames,
            latent_height,
            latent_width,
            self.transformer_spatial_patch_size,
            self.transformer_temporal_patch_size,
        )

        if offload:
            self._offload(self.transformer)

        batch_size = latents.shape[0]

        if not self.vae:
            self.load_component_by_type("vae")

        self.to_device(self.vae)

        if upsample_latents:
            latents = self.upsample_latents(latents, **upsample_kwargs)

        latents = self.vae.denormalize_latents(latents)

        latents = latents.to(dtype)

        if return_latents:
            return latents

        if not self.vae.config.timestep_conditioning:
            timestep = None
        else:
            noise = randn_tensor(
                latents.shape,
                generator=generator,
                device=self.device,
                dtype=latents.dtype,
            )
            if not isinstance(decode_timestep, list):
                decode_timestep = [decode_timestep] * batch_size
            if decode_noise_scale is None:
                decode_noise_scale = decode_timestep
            elif not isinstance(decode_noise_scale, list):
                decode_noise_scale = [decode_noise_scale] * batch_size
            timestep = torch.tensor(
                decode_timestep, device=self.device, dtype=latents.dtype
            )
            decode_noise_scale = torch.tensor(
                decode_noise_scale, device=self.device, dtype=latents.dtype
            )[:, None, None, None, None]
            latents = (1 - decode_noise_scale) * latents + decode_noise_scale * noise

        decoded_video = self.vae.decode(latents, timestep, return_dict=False)[0]
        video = self._tensor_to_frames(decoded_video)

        if offload:
            self._offload(self.vae)

        return video

