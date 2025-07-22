import torch


class MochiBaseEngine:
    """Base class for Mochi engine implementations containing common functionality"""

    def __init__(self, main_engine):
        self.main_engine = main_engine
        # Delegate common properties to the main engine
        self.device = main_engine.device
        self.logger = main_engine.logger
        self.vae_spatial_scale_factor = main_engine.vae_spatial_scale_factor
        self.vae_temporal_scale_factor = main_engine.vae_temporal_scale_factor
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

    def denoise(self, *args, **kwargs):
        """Denoise function"""
        return self.main_engine.denoise(*args, **kwargs)

    def vae_decode(
        self, latents: torch.Tensor, offload: bool = False, dtype: torch.dtype = None
    ):
        if self.vae is None:
            self.load_component_by_type("vae")
        self.to_device(self.vae)

        # unscale/denormalize the latents
        has_latents_mean = (
            hasattr(self.vae.config, "latents_mean")
            and self.vae.config.latents_mean is not None
        )
        has_latents_std = (
            hasattr(self.vae.config, "latents_std")
            and self.vae.config.latents_std is not None
        )
        if has_latents_mean and has_latents_std:
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.num_channels_latents, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents_std = (
                torch.tensor(self.vae.config.latents_std)
                .view(1, self.num_channels_latents, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents = (
                latents * latents_std / self.vae.config.scaling_factor + latents_mean
            )
        else:
            latents = latents / self.vae.config.scaling_factor

        video = self.vae.decode(latents.to(self.vae.dtype), return_dict=False)[0]

        if offload:
            self._offload(self.vae)

        return video.to(dtype=dtype)

    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        num_frames,
        dtype,
        device,
        generator,
        latents=None,
    ):
        height = height // self.vae_spatial_scale_factor
        width = width // self.vae_spatial_scale_factor
        num_frames = (num_frames - 1) // self.vae_temporal_scale_factor + 1

        shape = (batch_size, num_channels_latents, num_frames, height, width)

        if latents is not None:
            return latents.to(device=device, dtype=dtype)

        from diffusers.utils.torch_utils import randn_tensor

        latents = randn_tensor(
            shape, generator=generator, device=device, dtype=torch.float32
        )
        latents = latents.to(dtype)
        return latents
