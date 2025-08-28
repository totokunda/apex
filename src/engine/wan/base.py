import torch
from PIL import Image
import numpy as np
import torchvision.transforms.functional as F
import torch.nn.functional as F
import math
from torchvision import transforms


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

    @text_encoder.setter
    def text_encoder(self, text_encoder):
        self.main_engine.text_encoder = text_encoder

    @property
    def transformer(self):
        return self.main_engine.transformer

    @transformer.setter
    def transformer(self, transformer):
        self.main_engine.transformer = transformer

    @property
    def scheduler(self):
        return self.main_engine.scheduler

    @scheduler.setter
    def scheduler(self, scheduler):
        self.main_engine.scheduler = scheduler

    @property
    def vae(self):
        return self.main_engine.vae

    @vae.setter
    def vae(self, vae):
        self.main_engine.vae = vae

    @property
    def helpers(self):
        return self.main_engine.helpers

    @property
    def component_dtypes(self):
        return self.main_engine.component_dtypes

    def load_component_by_type(self, component_type: str):
        """Load a component by type"""
        return self.main_engine.load_component_by_type(component_type)

    def load_config_by_type(self, component_type: str):
        """Load a component by type and config"""
        return self.main_engine.load_config_by_type(component_type)
    
    def load_config_by_name(self, component_name: str):
        """Load a component by name and config"""
        return self.main_engine.load_config_by_name(component_name)

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

    def _tensor_to_frames(self, *args, **kwargs):
        """Convert torch.tensor to list of PIL images or np.ndarray"""
        return self.main_engine._tensor_to_frames(*args, **kwargs)

    def _tensor_to_frame(self, *args, **kwargs):
        """Convert torch.tensor to PIL image"""
        return self.main_engine._tensor_to_frame(*args, **kwargs)

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
