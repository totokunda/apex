import torch
from typing import Dict, Any, Callable, List, Union, Optional
from PIL import Image
import numpy as np


class HunyuanBaseEngine:
    """Base class for Hunyuan engine implementations containing common functionality"""

    def __init__(self, main_engine):
        self.main_engine = main_engine
        # Delegate common properties to the main engine
        self.device = main_engine.device
        self.logger = main_engine.logger
        self.vae_scale_factor_temporal = main_engine.vae_scale_factor_temporal
        self.vae_scale_factor_spatial = main_engine.vae_scale_factor_spatial
        self.num_channels_latents = main_engine.num_channels_latents
        self.video_processor = main_engine.video_processor
        self.llama_text_encoder = main_engine.llama_text_encoder

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

    def _calculate_shift(self, *args, **kwargs):
        """Calculate shift parameter for timestep scheduling"""
        return self.main_engine._calculate_shift(*args, **kwargs)

    def _soft_append(
        self, history: torch.Tensor, current: torch.Tensor, overlap: int = 0
    ):
        """Soft append with blending for framepack generation"""
        if overlap <= 0:
            return torch.cat([history, current], dim=2)

        assert (
            history.shape[2] >= overlap
        ), f"Current length ({history.shape[2]}) must be >= overlap ({overlap})"
        assert (
            current.shape[2] >= overlap
        ), f"History length ({current.shape[2]}) must be >= overlap ({overlap})"

        weights = torch.linspace(
            1, 0, overlap, dtype=history.dtype, device=history.device
        ).view(1, 1, -1, 1, 1)
        blended = (
            weights * history[:, :, -overlap:] + (1 - weights) * current[:, :, :overlap]
        )
        output = torch.cat(
            [history[:, :, :-overlap], blended, current[:, :, overlap:]], dim=2
        )

        return output.to(history)

    def _encode_prompt(
        self,
        prompt: Union[str, List[str]],
        prompt_2: Union[str, List[str], None] = None,
        image: Union[
            Image.Image, List[Image.Image], str, np.ndarray, torch.Tensor, None
        ] = None,
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = 256,
        dtype: Optional[torch.dtype] = None,
        image_embed_interleave: int = 2,
        hyavatar: bool = False,
        **kwargs,
    ):
        """Encode prompts using both LLaMA and CLIP text encoders"""
        if not self.text_encoder:
            self.load_component_by_type("text_encoder")

        self.to_device(self.text_encoder)

        if not "hunyuan.llama" in self.preprocessors:
            self.load_preprocessor_by_type("hunyuan.llama")
        self.to_device(self.preprocessors["hunyuan.llama"])

        if self.llama_text_encoder is None:
            self.llama_text_encoder = self.preprocessors["hunyuan.llama"]

        if isinstance(prompt, str):
            prompt = [prompt]

        if isinstance(prompt_2, str):
            prompt_2 = [prompt_2]

        if prompt_2 is None:
            prompt_2 = prompt

        prompt_embeds, prompt_attention_mask = self.llama_text_encoder(
            prompt,
            image=image,
            max_sequence_length=max_sequence_length,
            pad_to_max_length=True,
            num_videos_per_prompt=num_videos_per_prompt,
            dtype=dtype,
            image_embed_interleave=image_embed_interleave,
            hyavatar=hyavatar,
        )

        pooled_prompt_embeds = self.text_encoder.encode(
            prompt_2,
            max_sequence_length=77,
            pad_to_max_length=True,
            use_mask_in_input=False,
            use_position_ids=True,
            num_videos_per_prompt=num_videos_per_prompt,
            dtype=dtype,
            **kwargs,
        )

        return pooled_prompt_embeds, prompt_embeds, prompt_attention_mask

    def rescale_noise_cfg(self, noise_cfg, noise_pred_text, guidance_rescale=0.0):
        """
        Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
        Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
        """
        std_text = noise_pred_text.std(
            dim=list(range(1, noise_pred_text.ndim)), keepdim=True
        )
        std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
        # rescale the results from guidance (fixes overexposure)
        noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
        # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
        noise_cfg = (
            guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
        )
        return noise_cfg
