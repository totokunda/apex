import torch
from diffusers.video_processor import VideoProcessor
from enum import Enum, auto
from typing import List

from src.engine.wan_engine import BaseEngine, OffloadMixin
from src.engine.denoise.ltx_denoise import LTXDenoise, DenoiseType
from src.ui.nodes import UINode
from typing import Dict, Any, Callable
import math
from PIL import Image
import numpy as np
from typing import Union
from typing import Optional, Tuple
from diffusers.models.autoencoders import AutoencoderKL
from diffusers.utils.torch_utils import randn_tensor
import inspect
from src.mixins.loader_mixin import LoaderMixin

class ModelType(Enum):
    T2V = "t2v"  # text to video
    I2V = "i2v"  # image to video
    CONTROL = "control"

def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    r"""
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError(
            "Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values"
        )
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accept_sigmas:
            scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
            timesteps = scheduler.timesteps
        else:
            scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
            timesteps = scheduler.timesteps
            num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


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


class LTXEngine(BaseEngine, OffloadMixin, LTXDenoise):
    def __init__(
        self,
        yaml_path: str,
        model_type: ModelType = ModelType.T2V,
        **kwargs,
    ):
        super().__init__(yaml_path, **kwargs)

        self.model_type = model_type
        if self.model_type == ModelType.CONTROL:
            self.denoise_type = DenoiseType.CONDITION
        elif self.model_type == ModelType.T2V:
            self.denoise_type = DenoiseType.T2V
        elif self.model_type == ModelType.I2V:
            self.denoise_type = DenoiseType.I2V
        else:
            raise ValueError(f"Model type {self.model_type} not supported")

        self.vae_scale_factor_spatial = (
            self.vae.spatial_compression_ratio
            if getattr(self, "vae", None) is not None
            else 32
        )

        self.vae_scale_factor_temporal = (
            self.vae.temporal_compression_ratio
            if getattr(self, "vae", None) is not None
            else 8
        )

        self.transformer_spatial_patch_size = (
            self.transformer.config.patch_size
            if getattr(self, "transformer", None) is not None
            else 1
        )

        self.transformer_temporal_patch_size = (
            self.transformer.config.patch_size_t
            if getattr(self, "transformer") is not None
            else 1
        )

        self.video_processor = VideoProcessor(
            vae_scale_factor=self.vae_scale_factor_spatial
        )

        self.num_channels_latents: int = (
            self.vae.config.get("latent_channels", 128) if self.vae is not None else 128
        )

    def run(self, *args, input_nodes: List[UINode] = None, **kwargs):
        default_kwargs = self._get_default_kwargs("run")
        preprocessed_kwargs = self._preprocess_kwargs(input_nodes, **kwargs)
        final_kwargs = {**default_kwargs, **preprocessed_kwargs}

        if self.model_type == ModelType.T2V:
            return self.t2v_run(**final_kwargs)
        elif self.model_type == ModelType.I2V:
            return self.i2v_run(**final_kwargs)
        elif self.model_type == ModelType.CONTROL:
            return self.control_run(**final_kwargs)
        else:
            raise ValueError(f"Model type {self.model_type} not supported")

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

        latents = latents.to(self.vae.dtype)
        decoded_video = self.vae.decode(latents, timestep, return_dict=False)[0]
        video = self._postprocess(decoded_video)

        if offload:
            self._offload(self.vae)

        return video

    def t2v_run(
        self,
        prompt: List[str] | str,
        negative_prompt: List[str] | str = None,
        height: int = 480,
        width: int = 832,
        duration: int | str = 25,
        num_inference_steps: int = 30,
        num_videos: int = 1,
        seed: int | None = None,
        fps: int = 25,
        use_cfg_guidance: bool = True,
        text_encoder_kwargs: Dict[str, Any] = {},
        guidance_scale: float = 3.0,
        guidance_rescale: float = 0.0,
        offload: bool = True,
        render_on_step: bool = False,
        generator: torch.Generator | None = None,
        timesteps: List[int] | None = None,
        render_on_step_callback: Callable = None,
        attention_kwargs: Dict[str, Any] = {},
        return_latents: bool = False,
        decode_timestep: Union[float, List[float]] = 0.0,
        decode_noise_scale: Optional[Union[float, List[float]]] = None,
        **kwargs,
    ):

        if not self.text_encoder:
            self.load_component_by_type("text_encoder")

        self.to_device(self.text_encoder)
        

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

        if not self.transformer:
            self.load_component_by_type("transformer")

        self.to_device(self.transformer)

        transformer_dtype = self.component_dtypes["transformer"]

        latents, generator = self._get_latents(
            height=height,
            width=width,
            duration=duration,
            fps=fps,
            num_videos=num_videos,
            seed=seed,
            dtype=torch.float32,
            generator=generator,
            return_generator=True,
        )
        
        latents = self._pack_latents(
            latents,
            self.transformer_spatial_patch_size,
            self.transformer_temporal_patch_size,
        )

        if not self.scheduler:
            self.load_component_by_type("scheduler")

        scheduler = self.scheduler

        self.to_device(scheduler)
        num_frames = self._parse_num_frames(duration, fps)

        latent_num_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        latent_height = height // self.vae_scale_factor_spatial
        latent_width = width // self.vae_scale_factor_spatial
        video_sequence_length = latent_num_frames * latent_height * latent_width

        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        
        mu = calculate_shift(
            video_sequence_length,
            scheduler.config.get("base_image_seq_len", 256),
            scheduler.config.get("max_image_seq_len", 4096),
            scheduler.config.get("base_shift", 0.5),
            scheduler.config.get("max_shift", 1.15),
        )

        timesteps, num_inference_steps = retrieve_timesteps(
            scheduler,
            num_inference_steps=num_inference_steps,
            device=self.device,
            timesteps=timesteps,
            sigmas=sigmas if not timesteps else None,
            mu=mu,
        )

        prompt_embeds = prompt_embeds.to(device=self.device, dtype=transformer_dtype)
        prompt_attention_mask = prompt_attention_mask.to(
            device=self.device
        )

        num_warmup_steps = max(
            len(timesteps) - num_inference_steps * scheduler.order, 0
        )

        # 6. Prepare micro-conditions
        rope_interpolation_scale = (
            self.vae_scale_factor_temporal / fps,
            self.vae_scale_factor_spatial,
            self.vae_scale_factor_spatial,
        )

        latents = self.denoise(
            latents=latents,
            timesteps=timesteps,
            num_warmup_steps=num_warmup_steps,
            num_inference_steps=num_inference_steps,
            num_videos=num_videos,
            seed=seed,
            fps=fps,
            prompt_embeds=prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            latent_num_frames=latent_num_frames,
            latent_height=latent_height,
            latent_width=latent_width,
            rope_interpolation_scale=rope_interpolation_scale,
            attention_kwargs=attention_kwargs,
            guidance_rescale=guidance_rescale,
            use_cfg_guidance=use_cfg_guidance,
            guidance_scale=guidance_scale,
            render_on_step=render_on_step,
            scheduler=scheduler,
            transformer_dtype=transformer_dtype,
            render_on_step_callback=render_on_step_callback,
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
            dtype=prompt_embeds.dtype,
        )

    def i2v_run(
        self,
        image: Union[Image.Image, str, np.ndarray, torch.Tensor],
        prompt: List[str] | str,
        negative_prompt: List[str] | str = None,
        height: int = 480,
        width: int = 832,
        duration: int | str = 16,
        num_inference_steps: int = 30,
        num_videos: int = 1,
        seed: int | None = None,
        fps: int = 25,
        use_cfg_guidance: bool = True,
        text_encoder_kwargs: Dict[str, Any] = {},
        guidance_scale: float = 5.0,
        guidance_rescale: float = 0.0,
        offload: bool = True,
        render_on_step: bool = False,
        generator: torch.Generator | None = None,
        timesteps: List[int] | None = None,
        render_on_step_callback: Callable = None,
        attention_kwargs: Dict[str, Any] = {},
        return_latents: bool = False,
        decode_timestep: Union[float, List[float]] = 0.0,
        decode_noise_scale: Optional[Union[float, List[float]]] = None,
        **kwargs,
    ):
        if not self.text_encoder:
            self.load_component_by_type("text_encoder")

        self.to_device(self.text_encoder)
        
        if seed is not None and generator is None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        elif seed is not None and generator is not None:
            self.logger.warning("Both seed and generator are provided. Ignoring seed and using generator.")
            seed = None

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

        if not self.transformer:
            self.load_component_by_type("transformer")

        self.to_device(self.transformer)
        transformer_dtype = self.component_dtypes["transformer"]

        if image is not None:
            loaded_image = self._load_image(image)
            # make height divisible by vae_scale_factor_spatial
            height = height // self.vae_scale_factor_spatial * self.vae_scale_factor_spatial
            width = width // self.vae_scale_factor_spatial * self.vae_scale_factor_spatial

            prepocessed_image = self.video_processor.preprocess(
                loaded_image, height=height, width=width
            )
            prepocessed_image = prepocessed_image.to(
                device=self.device, dtype=transformer_dtype
            ).unsqueeze(2)
        
        num_frames = self._parse_num_frames(duration, fps)
        latent_num_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        latent_height = height // self.vae_scale_factor_spatial
        latent_width = width // self.vae_scale_factor_spatial
        video_sequence_length = latent_num_frames * latent_height * latent_width
        batch_size = num_videos
        
        init_latents = self.vae_encode(
            prepocessed_image, sample_generator=generator, sample_mode="sample", dtype=torch.float32, offload=offload
        ).repeat(1, 1, latent_num_frames, 1, 1)
        
        conditioning_mask = torch.zeros(
            batch_size,
            1,
            latent_num_frames,
            latent_height,
            latent_width,
            device=self.device,
            dtype=torch.float32,
        )
        
        conditioning_mask[:, :, 0] = 1.0
        
        noise_latents = self._get_latents(
            height=height,
            width=width,
            duration=latent_num_frames,
            fps=fps,
            num_videos=num_videos,
            dtype=torch.float32,
            seed=seed,
            generator=generator,
            parse_frames=False,
        )
        

        latents = init_latents * conditioning_mask + noise_latents * (
            1 - conditioning_mask
        )
        
        conditioning_mask = self._pack_latents(
            conditioning_mask,
            self.transformer_spatial_patch_size,
            self.transformer_temporal_patch_size,
        ).squeeze(-1)

        latents = self._pack_latents(
            latents,
            self.transformer_spatial_patch_size,
            self.transformer_temporal_patch_size,
        )

        prompt_embeds = prompt_embeds.to(device=self.device, dtype=transformer_dtype)
        prompt_attention_mask = prompt_attention_mask.to(
            device=self.device
        )
        
        if not self.scheduler:
            self.load_component_by_type("scheduler")
        
        self.to_device(self.scheduler)
            
        scheduler = self.scheduler
        
        if use_cfg_guidance:
            conditioning_mask = torch.cat([conditioning_mask, conditioning_mask])

        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        mu = calculate_shift(
            video_sequence_length,
            scheduler.config.get("base_image_seq_len", 256),
            scheduler.config.get("max_image_seq_len", 4096),
            scheduler.config.get("base_shift", 0.5),
            scheduler.config.get("max_shift", 1.15),
        )

        timesteps, num_inference_steps = retrieve_timesteps(
            scheduler,
            num_inference_steps,
            self.device,
            timesteps,
            mu=mu,  
            sigmas=sigmas if not timesteps else None,
        )

        num_warmup_steps = max(
            len(timesteps) - num_inference_steps * scheduler.order, 0
        )
        
        # 6. Prepare micro-conditions
        rope_interpolation_scale = (
            self.vae_scale_factor_temporal / fps,
            self.vae_scale_factor_spatial,
            self.vae_scale_factor_spatial,
        )

        latents = self.denoise(
            latents=latents,
            timesteps=timesteps,
            num_warmup_steps=num_warmup_steps,
            num_inference_steps=num_inference_steps,
            num_videos=num_videos,
            seed=seed,
            fps=fps,
            prompt_embeds=prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            latent_num_frames=latent_num_frames,
            latent_height=latent_height,
            latent_width=latent_width,
            rope_interpolation_scale=rope_interpolation_scale,
            attention_kwargs=attention_kwargs,
            guidance_rescale=guidance_rescale,
            use_cfg_guidance=use_cfg_guidance,
            guidance_scale=guidance_scale,
            render_on_step=render_on_step,
            transformer_dtype=transformer_dtype,
            render_on_step_callback=render_on_step_callback,
            conditioning_mask=conditioning_mask,
            scheduler=scheduler,
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
        )

    def control_run(
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

        if not self.text_encoder:
            self.load_component_by_type("text_encoder")

        self.to_device(self.text_encoder)
        text_encoder_kwargs['max_sequence_length'] = 256
        text_encoder_kwargs['use_mask_in_input'] = True

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
        timesteps, num_inference_steps = retrieve_timesteps(
            scheduler, num_inference_steps, self.device, timesteps
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
                condition_latents = self.vae_encode(data, sample_generator=generator, offload=offload, sample_mode="sample", dtype=torch.float32)
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
            conditioning_mask = torch.cat([*extra_conditioning_mask, conditioning_mask], dim=1)
        
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
        )

if __name__ == "__main__":
    from diffusers.utils import export_to_video
    from PIL import Image

    engine = LTXEngine(
        yaml_path="manifest/ltx_x2v_13b.yml",
        model_type=ModelType.CONTROL,
        save_path="/mnt/localssd",  # Change this to your desired save path
        components_to_load=["transformer", "vae"]
    )

    image = Image.open('/path/to/image.jpg')
    prompt="A quiet, romantic indoor scene. A couple stands close together in a softly lit white room, their bodies turned inward, lost in each other. The woman gently places her arms around the man’s shoulders, while he wraps his arms around her waist. Slowly, they lean in and share a tender kiss. The camera circles around them as they embrace, highlighting the subtle shifts in their posture and the stillness of the moment. Her long hair moves softly as they kiss, and their connection is intimate, warm, and calm. Time seems to pause. As the kiss gently fades, they remain close—foreheads touching, eyes closed, smiling softly—surrounded by silence, affection, and peace."
    negative_prompt="worst quality, inconsistent motion, blurry, jittery, distorted"

    height = 480
    width = 832
    print(f"height: {height}, width: {width}")

    conditions = [
        LTXVideoCondition(image=image, strength=1.0, frame_index=0)
    ]

    video = engine.run(
        conditions=conditions,
        height=height,
        width=width,
        prompt=prompt,
        negative_prompt=negative_prompt,
        use_cfg_guidance=True,
        duration="121f",
        num_videos=1,
        guidance_scale=3.0,
        num_inference_steps=30,
        generator=torch.Generator(device="cuda").manual_seed(42),
    )

    export_to_video(video[0], "control_ltx2v.mp4", fps=24, quality=8)
