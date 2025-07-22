import torch
from typing import Dict, Any, Callable, List, Union, Optional
from PIL import Image
import numpy as np
import torchvision.transforms.functional as TF
import torch.nn.functional as F


class WanBaseEngine:
    """Base class for WAN engine implementations containing common functionality"""

    def __init__(self, main_engine):
        self.main_engine = main_engine
        # Delegate common properties to the main engine
        self.device = main_engine.device
        self.logger = main_engine.logger
        self.vae_scale_factor_temporal = main_engine.vae_scale_factor_temporal
        self.vae_scale_factor_spatial = main_engine.vae_scale_factor_spatial
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

    def _prepare_fun_control_latents(
        self, control, dtype=torch.float32, generator: torch.Generator | None = None
    ):
        """Prepare control latents for FUN implementation"""
        # resize the control to latents shape as we concatenate the control to the latents
        # we do that before converting to dtype to avoid breaking in case we're using cpu_offload
        # and half precision

        control = control.to(device=self.device, dtype=dtype)
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

    def _apply_color_correction(
        self, videos: torch.Tensor, reference_image: torch.Tensor, strength: float
    ) -> torch.Tensor:
        """Apply color correction to match reference image."""
        # Simple color matching - can be enhanced with more sophisticated methods
        # This is a placeholder implementation
        return videos
