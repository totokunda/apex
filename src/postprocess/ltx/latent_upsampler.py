import torch
import torch.nn.functional as F
from typing import Dict, Any, Optional
from diffusers.models.autoencoders import AutoencoderKL
from diffusers.pipelines.ltx.modeling_latent_upsampler import LTXLatentUpsamplerModel
from src.postprocess.base import Postprocessor, postprocessor_registry
from src.utils.cache_utils import empty_cache


@postprocessor_registry("ltx.latent_upsampler")
class LatentUpsamplerPostprocessor(Postprocessor):
    def __init__(self, engine, **kwargs):
        super().__init__(engine, **kwargs)
        
        # Get configuration from component_conf
        self.config = self.component_conf
        self.device = engine.device
        self.component_dtypes = getattr(engine, 'component_dtypes', {})
        self.component_load_dtypes = getattr(engine, 'component_load_dtypes', {})
        
        # Default dtype for latent upsampler
        self.dtype = self.component_dtypes.get('latent_upsampler', torch.float16)
        self.load_dtype = self.component_load_dtypes.get('latent_upsampler', torch.float16)
        
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
                        "type": "latent_upsampler"
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
                device=self.device, 
                dtype=self.dtype
            )
            
            self.engine.logger.info("Latent upsampler loaded successfully")
            empty_cache()
            
        except Exception as e:
            self.engine.logger.error(f"Failed to load latent upsampler: {e}")
            raise

    def _prepare_latents(
        self, 
        latents: torch.Tensor,
        height: Optional[int] = None,
        width: Optional[int] = None,
    ) -> torch.Tensor:
        """Prepare latents for upsampling"""
        # Ensure latents are on correct device and dtype
        latents = latents.to(device=self.device, dtype=self.dtype)
        
        # Get dimensions
        batch_size, channels, frames, latent_height, latent_width = latents.shape
        
        # Calculate target dimensions if not provided
        if height is None:
            height = latent_height * 2  # Default 2x upsampling
        if width is None:
            width = latent_width * 2   # Default 2x upsampling
        
        return latents, height, width

    def _upsample_latents(
        self,
        latents: torch.Tensor,
        height: int,
        width: int,
        num_inference_steps: int = 4,
        guidance_scale: float = 3.0,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Perform latent upsampling using the loaded model"""
        
        batch_size, channels, frames, latent_height, latent_width = latents.shape
        
        # Calculate scale factors
        height_scale_factor = height // latent_height
        width_scale_factor = width // latent_width
        
        # Create noise for upsampling process
        if generator is None:
            generator = torch.Generator(device=self.device)
        
        # Initialize upsampled latents with interpolation
        upsampled_shape = (batch_size, channels, frames, height, width)
        upsampled_latents = F.interpolate(
            latents.flatten(0, 2),  # Flatten batch and frame dimensions
            size=(height, width),
            mode='bilinear',
            align_corners=False
        ).view(upsampled_shape)
        
        # Add noise for the upsampling process
        noise = torch.randn(
            upsampled_shape,
            generator=generator,
            device=self.device,
            dtype=self.dtype
        )
        
        # Perform denoising steps using the latent upsampler
        with torch.no_grad():
            for step in range(num_inference_steps):
                # Calculate timestep
                timestep = torch.tensor(
                    [1000 - (step * 1000 // num_inference_steps)], 
                    device=self.device
                )
                
                # Prepare model input
                model_input = upsampled_latents
                
                # Get noise prediction from the upsampler model
                noise_pred = self.latent_upsampler(
                    model_input,
                    timestep,
                    encoder_hidden_states=latents,  # Use original latents as conditioning
                    return_dict=False
                )[0]
                
                # Apply guidance if scale > 1
                if guidance_scale > 1.0:
                    # For simplicity, we'll use the noise prediction as-is
                    # In a full implementation, you might want to split conditional/unconditional
                    pass
                
                # Update latents (simplified scheduler step)
                alpha = 1.0 - (step + 1) / num_inference_steps
                upsampled_latents = alpha * upsampled_latents + (1 - alpha) * (upsampled_latents - noise_pred)
        
        return upsampled_latents

    @torch.no_grad()
    def __call__(
        self, 
        latents: torch.Tensor, 
        vae: Optional[AutoencoderKL] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 4,
        guidance_scale: float = 3.0,
        generator: Optional[torch.Generator] = None,
        return_latents: bool = True,
        **kwargs
    ) -> torch.Tensor:
        """
        Upsample latents using the LTX latent upsampler
        
        Args:
            latents: Input latents to upsample
            vae: VAE model (optional, can use engine's VAE)
            height: Target height for upsampling
            width: Target width for upsampling
            num_inference_steps: Number of denoising steps
            guidance_scale: Guidance scale for upsampling
            generator: Random generator for reproducibility
            return_latents: Whether to return latents or decoded images
            **kwargs: Additional arguments
            
        Returns:
            Upsampled latents or decoded images
        """
        
        # Ensure the latent upsampler is loaded
        if self.latent_upsampler is None:
            self._load_latent_upsampler()
        
        # Move latent upsampler to device if needed
        if self.latent_upsampler.device != self.device:
            self.latent_upsampler = self.latent_upsampler.to(self.device)
        
        # Prepare latents
        prepared_latents, target_height, target_width = self._prepare_latents(
            latents, height, width
        )
        
        self.engine.logger.info(
            f"Upsampling latents from {prepared_latents.shape} to "
            f"[{prepared_latents.shape[0]}, {prepared_latents.shape[1]}, "
            f"{prepared_latents.shape[2]}, {target_height}, {target_width}]"
        )
        
        # Perform upsampling
        upsampled_latents = self._upsample_latents(
            prepared_latents,
            target_height,
            target_width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )
        
        if return_latents:
            return upsampled_latents
        
        # Decode to images if requested
        if vae is None:
            # Use engine's VAE if available
            vae = getattr(self.engine, 'vae', None)
            if vae is None:
                self.engine.logger.warning("No VAE available for decoding, returning latents")
                return upsampled_latents
        
        # Decode latents to images
        self.engine.logger.info("Decoding upsampled latents to images")
        
        # Ensure VAE is on correct device
        if hasattr(vae, 'to'):
            vae = vae.to(self.device)
        
        # Decode in chunks to avoid memory issues
        batch_size, channels, frames, height, width = upsampled_latents.shape
        decoded_frames = []
        
        for i in range(frames):
            frame_latents = upsampled_latents[:, :, i:i+1, :, :]  # Keep frame dimension
            with torch.no_grad():
                decoded_frame = vae.decode(frame_latents, return_dict=False)[0]
                decoded_frames.append(decoded_frame)
        
        # Concatenate decoded frames
        decoded_video = torch.cat(decoded_frames, dim=2)
        
        return decoded_video

    def to(self, device: torch.device):
        """Move the postprocessor to a device"""
        self.device = device
        if self.latent_upsampler is not None:
            self.latent_upsampler = self.latent_upsampler.to(device)
        return self

    def __str__(self):
        return f"LatentUpsamplerPostprocessor(device={self.device}, dtype={self.dtype})"

    def __repr__(self):
        return self.__str__()
