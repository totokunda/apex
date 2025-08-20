import torch
from typing import Literal
from diffusers.pipelines.ltx.modeling_latent_upsampler import LTXLatentUpsamplerModel
from src.postprocess.base import BasePostprocessor, PostprocessorCategory, postprocessor_registry
from src.utils.cache import empty_cache
import numpy as np
from PIL import Image


@postprocessor_registry("ltx.latent_upsampler")
class LatentUpsamplerPostprocessor(BasePostprocessor):
    def __init__(self, engine, **kwargs):
        super().__init__(engine, PostprocessorCategory.UPSCALER, **kwargs)

        # Get configuration from component_conf
        self.config = self.component_conf
        self.device = engine.device
        self.component_dtypes = getattr(engine, "component_dtypes", {})
        self.component_load_dtypes = getattr(engine, "component_load_dtypes", {})

        # Default dtype for latent upsampler
        self.dtype = self.component_dtypes.get("latent_upsampler", torch.bfloat16)
        self.load_dtype = self.component_load_dtypes.get(
            "latent_upsampler", torch.bfloat16
        )

        # Initialize the latent upsampler model
        self.latent_upsampler = None
        self._load_latent_upsampler()

    def _load_latent_upsampler(self):
        """Load the latent upsampler model following engine patterns"""
        try:
            # Check if model_path is provided
            model_path = self.config.get("model_path")
            if not model_path:
                raise ValueError("model_path is required for latent upsampler")

            # Get configuration
            config_path = self.config.get("config_path")
            upsampler_config = self.config.get("config", {})

            if config_path:
                fetched_config = self.engine.fetch_config(config_path)
                upsampler_config = {**fetched_config, **upsampler_config}

            self.engine.logger.info(f"Loading latent upsampler from {model_path}")

            # Load model using proper loading mechanics
            if upsampler_config:
                # Load with custom config
                self.latent_upsampler = self._load_model(
                    component={
                        "base": "LTXLatentUpsamplerModel",
                        "model_path": model_path,
                        "config": upsampler_config,
                        "type": "latent_upsampler",
                    },
                    module_name="diffusers.pipelines.ltx.modeling_latent_upsampler",
                    load_dtype=self.load_dtype,
                )
            else:
                # Load using from_pretrained
                self.latent_upsampler = LTXLatentUpsamplerModel.from_pretrained(
                    model_path, torch_dtype=self.load_dtype
                )

            # Move to device and set dtype
            self.latent_upsampler = self.latent_upsampler.to(
                device=self.device, dtype=self.dtype
            )

            self.engine.logger.info("Latent upsampler loaded successfully")
            empty_cache()

        except Exception as e:
            self.engine.logger.error(f"Failed to load latent upsampler: {e}")
            raise

    def _upsample_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Perform latent upsampling using the loaded model.

        Note: The LTX latent upsampler expects denormalized latents and returns upsampled latents directly.
        """

        # Match the upsampler's parameter dtype
        try:
            upsampler_dtype = next(self.latent_upsampler.parameters()).dtype
        except Exception:
            upsampler_dtype = self.dtype

        latents = latents.to(device=self.device, dtype=upsampler_dtype)

        with torch.no_grad():
            upsampled_latents = self.latent_upsampler(latents)

        return upsampled_latents

    def _adain_filter_latent(
        self,
        latents: torch.Tensor,
        reference_latents: torch.Tensor,
        factor: float = 1.0,
    ) -> torch.Tensor:
        result = latents.clone()
        for i in range(latents.size(0)):
            for c in range(latents.size(1)):
                r_sd, r_mean = torch.std_mean(reference_latents[i, c], dim=None)
                i_sd, i_mean = torch.std_mean(result[i, c], dim=None)
                result[i, c] = ((result[i, c] - i_mean) / i_sd) * r_sd + r_mean
        result = torch.lerp(latents, result, factor)
        return result

    @torch.no_grad()
    def __call__(
        self,
        latents: torch.Tensor | None = None,
        video: str | list[str] | list[Image.Image] | list[np.ndarray] | None = None,
        return_latents: bool = True,
        output_type: Literal["pil", "np"] = "pil",
        **kwargs,
    ) -> torch.Tensor:
        """
        Upsample latents using the LTX latent upsampler

        Args:
            latents: Input latents to upsample
            video: Video to upsample
            output_type: Type of output to return
            return_latents: Whether to return latents or decoded images
            **kwargs: Additional arguments

        Returns:
            Upsampled latents or decoded images
        """

        # Ensure the latent upsampler is loaded
        if self.latent_upsampler is None:
            self._load_latent_upsampler()

        # Optionally denormalize latents using VAE stats if available
        self.engine.load_component_by_type("vae")
        self.engine.to_device(self.engine.vae)

        if latents is not None and video is not None:
            raise ValueError("Either latents or video must be provided, not both")

        if latents is None and video is None:
            raise ValueError("Either latents or video must be provided")

        if latents is None:
            video = self.engine._load_video(video)
            video = self.engine.video_processor.preprocess_video(video)
            latents = self.engine.vae.encode(video, sample_mode="mode")

        prepared_latents = self.engine.vae.denormalize_latents(latents)

        # Perform upsampling via the upsampler model
        upsampled_latents = self._upsample_latents(prepared_latents)

        if return_latents:
            upsampled_latents = self.engine.vae.normalize_latents(upsampled_latents)
            return upsampled_latents

        # Optionally apply AdaIN in latent space prior to decoding
        adain_factor = kwargs.get("adain_factor", 0.0)
        if adain_factor and adain_factor > 0.0:
            # Use the denormalized input latents as the reference
            upsampled_latents = self._adain_filter_latent(
                upsampled_latents, prepared_latents, float(adain_factor)
            )

        # Decode latents to images
        self.engine.logger.info("Decoding upsampled latents to images")

        # If VAE supports timestep conditioning, add decoding noise and pass timestep
        timestep = None
        vae = self.engine.vae

        if getattr(getattr(vae, "config", object()), "timestep_conditioning", False):
            batch_size = upsampled_latents.shape[0]
            decode_timestep = kwargs.get("decode_timestep", 0.0)
            decode_noise_scale = kwargs.get("decode_noise_scale", None)
            if not isinstance(decode_timestep, list):
                decode_timestep = [float(decode_timestep)] * batch_size
            if decode_noise_scale is None:
                decode_noise_scale = decode_timestep
            elif not isinstance(decode_noise_scale, list):
                decode_noise_scale = [float(decode_noise_scale)] * batch_size
            timestep = torch.tensor(
                decode_timestep, device=self.device, dtype=upsampled_latents.dtype
            )
            scale = torch.tensor(
                decode_noise_scale, device=self.device, dtype=upsampled_latents.dtype
            )[:, None, None, None, None]
            noise = torch.randn_like(upsampled_latents)
            upsampled_latents = (1 - scale) * upsampled_latents + scale * noise

        # Decode in chunks to avoid memory issues
        decoded_video = self.engine.vae.decode(
            upsampled_latents, timestep=timestep, return_dict=False
        )[0]
        decoded_video = self.engine._postprocess(decoded_video)

        return decoded_video

    def __str__(self):
        return f"LatentUpsamplerPostprocessor(device={self.device}, dtype={self.dtype})"

    def __repr__(self):
        return self.__str__()
