import torch
from typing import Dict, Any, Callable, List, Union, Optional
from PIL import Image
import numpy as np
from diffusers.models.embeddings import get_3d_rotary_pos_embed
import inspect

class CogVideoBaseEngine:
    """Base class for CogVideo engine implementations containing common functionality"""

    def __init__(self, main_engine):
        self.main_engine = main_engine
        # Delegate common properties to the main engine
        self.device = main_engine.device
        self.logger = main_engine.logger
        self.vae_scale_factor_temporal = main_engine.vae_scale_factor_temporal
        self.vae_scale_factor_spatial = main_engine.vae_scale_factor_spatial
        self.vae_scaling_factor_image = main_engine.vae_scaling_factor_image
        self.num_channels_latents = main_engine.num_channels_latents
        self.video_processor = main_engine.video_processor

    @property
    def text_encoder(self):
        return self.main_engine.text_encoder

    @property
    def transformer(self):
        return self.main_engine.transformer

    @property
    def scheduler(self):
        return self.main_engine.scheduler

    @property
    def vae(self):
        return self.main_engine.vae

    @property
    def preprocessors(self):
        return self.main_engine.preprocessors

    @property
    def component_dtypes(self):
        return self.main_engine.component_dtypes

    def load_component_by_type(self, component_type: str):
        """Load a component by type"""
        return self.main_engine.load_component_by_type(component_type)

    def load_preprocessor_by_type(self, preprocessor_type: str):
        """Load a preprocessor by type"""
        return self.main_engine.load_preprocessor_by_type(preprocessor_type)

    def to_device(self, component):
        """Move component to device"""
        return self.main_engine.to_device(component)

    def _offload(self, component):
        """Offload component"""
        return self.main_engine._offload(component)

    def _get_latents(self, *args, **kwargs):
        """Get latents"""
        return self.main_engine._get_latents(*args, **kwargs)

    def _get_timesteps(self, *args, **kwargs):
        """Get timesteps"""
        return self.main_engine._get_timesteps(*args, **kwargs)

    def _parse_num_frames(self, *args, **kwargs):
        """Parse number of frames"""
        return self.main_engine._parse_num_frames(*args, **kwargs)

    def _aspect_ratio_resize(self, *args, **kwargs):
        """Aspect ratio resize"""
        return self.main_engine._aspect_ratio_resize(*args, **kwargs)

    def _load_image(self, *args, **kwargs):
        """Load image"""
        return self.main_engine._load_image(*args, **kwargs)

    def _load_video(self, *args, **kwargs):
        """Load video"""
        return self.main_engine._load_video(*args, **kwargs)

    def _progress_bar(self, *args, **kwargs):
        """Progress bar context manager"""
        return self.main_engine._progress_bar(*args, **kwargs)

    def _postprocess(self, *args, **kwargs):
        """Postprocess video"""
        return self.main_engine._postprocess(*args, **kwargs)

    def vae_encode(self, *args, **kwargs):
        """VAE encode"""
        return self.main_engine.vae_encode(*args, **kwargs)

    def vae_decode(self, *args, **kwargs):
        """VAE decode"""
        return self.main_engine.vae_decode(*args, **kwargs)

    def denoise(self, *args, **kwargs):
        """Denoise function"""
        return self.main_engine.denoise(*args, **kwargs)

    def _encode_prompt(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Union[str, List[str]] = None,
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = 226,
        dtype: Optional[torch.dtype] = None,
        **kwargs,
    ):
        """Encode prompts using T5 text encoder"""
        if not self.text_encoder:
            self.load_component_by_type("text_encoder")

        self.to_device(self.text_encoder)

        if isinstance(prompt, str):
            prompt = [prompt]
            
        

        prompt_embeds = self.text_encoder.encode(
            prompt,
            max_sequence_length=max_sequence_length,
            pad_to_max_length=True,
            num_videos_per_prompt=num_videos_per_prompt,
            use_mask_in_input=False,
            pad_with_zero=False,
            dtype=dtype,
        )
        

        # Handle negative prompt
        negative_prompt_embeds = None
        if negative_prompt is not None:
            negative_prompt_embeds = self.text_encoder.encode(
                negative_prompt,
                max_sequence_length=max_sequence_length,
                pad_to_max_length=True,
                num_videos_per_prompt=num_videos_per_prompt,
                use_mask_in_input=False,
                pad_with_zero=False,
                dtype=dtype,
            )

        return prompt_embeds, negative_prompt_embeds

    def _prepare_rotary_positional_embeddings(
        self,
        height: int,
        width: int,
        num_frames: int,
        device: torch.device,
    ):
        """Prepare rotary positional embeddings for CogVideoX"""
        if not getattr(
            self.transformer.config, "use_rotary_positional_embeddings", False
        ):
            return None

        grid_height = height // (
            self.vae_scale_factor_spatial * self.transformer.config.patch_size
        )
        grid_width = width // (
            self.vae_scale_factor_spatial * self.transformer.config.patch_size
        )

        p = self.transformer.config.patch_size
        p_t = getattr(self.transformer.config, "patch_size_t", None)

        base_size_width = self.transformer.config.sample_width // p
        base_size_height = self.transformer.config.sample_height // p

        if p_t is None:
            # CogVideoX 1.0
            from thirdparty.diffusers.src.diffusers.pipelines.cogvideo.pipeline_cogvideox import (
                get_resize_crop_region_for_grid,
            )

            grid_crops_coords = get_resize_crop_region_for_grid(
                (grid_height, grid_width), base_size_width, base_size_height
            )
            freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
                embed_dim=self.transformer.config.attention_head_dim,
                crops_coords=grid_crops_coords,
                grid_size=(grid_height, grid_width),
                temporal_size=num_frames,
                device=device,
            )
        else:
            # CogVideoX 1.5
            base_num_frames = (num_frames + p_t - 1) // p_t

            freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
                embed_dim=self.transformer.config.attention_head_dim,
                crops_coords=None,
                grid_size=(grid_height, grid_width),
                temporal_size=base_num_frames,
                grid_type="slice",
                max_size=(base_size_height, base_size_width),
                device=device,
            )

        return freqs_cos, freqs_sin
    
    

    def _get_v2v_timesteps(
        self, num_inference_steps: int, timesteps: List[int], strength: float
    ):
        """Get timesteps for video-to-video generation based on strength"""
        # get the original timestep using init_timestep
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = timesteps[t_start * self.scheduler.order :]

        return timesteps, num_inference_steps - t_start

    def _prepare_v2v_latents(
        self,
        video: torch.Tensor,
        batch_size: int,
        num_channels_latents: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: Optional[torch.Generator] = None,
        timestep: Optional[torch.Tensor] = None,
    ):
        """Prepare latents for video-to-video generation"""
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        num_frames = (video.size(2) - 1) // self.vae_scale_factor_temporal + 1

        shape = (
            batch_size,
            num_frames,
            num_channels_latents,
            height // self.vae_scale_factor_spatial,
            width // self.vae_scale_factor_spatial,
        )

        # Encode video to latents using vae_encode
        if isinstance(generator, list):
            init_latents = [
                self.vae_encode(
                    video[i].unsqueeze(0),
                    sample_mode="sample",
                    sample_generator=generator[i],
                    dtype=dtype,
                )
                for i in range(batch_size)
            ]
        else:
            init_latents = [
                self.vae_encode(
                    vid.unsqueeze(0),
                    sample_mode="sample",
                    sample_generator=generator,
                    dtype=dtype,
                )
                for vid in video
            ]

        init_latents = torch.cat(init_latents, dim=0).permute(
            0, 2, 1, 3, 4
        )  # [B, F, C, H, W]

        # Add noise to the initial latents
        from diffusers.utils.torch_utils import randn_tensor

        noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        latents = self.scheduler.add_noise(init_latents, noise, timestep)

        # Scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents
    
    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://huggingface.co/papers/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def _retrieve_latents(
        self,
        encoder_output: torch.Tensor,
        generator: Optional[torch.Generator] = None,
        sample_mode: str = "sample",
    ):
        """Retrieve latents from encoder output"""
        if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
            return encoder_output.latent_dist.sample(generator)
        elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
            return encoder_output.latent_dist.mode()
        elif hasattr(encoder_output, "latents"):
            return encoder_output.latents
        else:
            raise AttributeError("Could not access latents of provided encoder_output")

    def _prepare_control_latents(
        self,
        mask: Optional[torch.Tensor] = None,
        masked_image: Optional[torch.Tensor] = None,
    ):
        """Prepare control latents for control video generation"""
        if mask is not None:
            masks = []
            for i in range(mask.size(0)):
                current_mask = mask[i].unsqueeze(0)
                current_mask = self.vae.encode(current_mask)[0]
                current_mask = current_mask.mode()
                masks.append(current_mask)
            mask = torch.cat(masks, dim=0)
            mask = mask * self.vae.config.scaling_factor

        if masked_image is not None:
            mask_pixel_values = []
            for i in range(masked_image.size(0)):
                mask_pixel_value = masked_image[i].unsqueeze(0)
                mask_pixel_value = self.vae.encode(mask_pixel_value)[0]
                mask_pixel_value = mask_pixel_value.mode()
                mask_pixel_values.append(mask_pixel_value)
            masked_image_latents = torch.cat(mask_pixel_values, dim=0)
            masked_image_latents = masked_image_latents * self.vae.config.scaling_factor
        else:
            masked_image_latents = None

        return mask, masked_image_latents
