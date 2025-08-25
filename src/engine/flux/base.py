import torch
from diffusers.utils.torch_utils import randn_tensor
from typing import Union, List, Optional, Dict, Any
from PIL import Image

class FluxBaseEngine:
    """Base class for Flux engine implementations containing common functionality"""

    def __init__(self, main_engine):
        self.main_engine = main_engine
        self.device = main_engine.device
        self.logger = main_engine.logger
        self.vae_scale_factor = main_engine.vae_scale_factor
        self.num_channels_latents = main_engine.num_channels_latents
        self.image_processor = main_engine.image_processor

    @property
    def text_encoder(self):
        return self.main_engine.text_encoder

    @property
    def text_encoder_2(self):
        return getattr(self.main_engine, "text_encoder_2", None)

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

    def load_component_by_name(self, component_name: str):
        """Load a component by name"""
        return self.main_engine.load_component_by_name(component_name)

    def load_preprocessor_by_type(self, preprocessor_type: str):
        """Load a preprocessor by type"""
        return self.main_engine.load_preprocessor_by_type(preprocessor_type)

    def to_device(self, component):
        """Move component to device"""
        return self.main_engine.to_device(component)

    def _offload(self, component):
        """Offload component"""
        return self.main_engine._offload(component)

    @staticmethod
    def _pack_latents(latents, batch_size, num_channels_latents, height, width):
        latents = latents.view(
            batch_size, num_channels_latents, height // 2, 2, width // 2, 2
        )
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(
            batch_size, (height // 2) * (width // 2), num_channels_latents * 4
        )

        return latents

    @staticmethod
    def _unpack_latents(latents, height, width, vae_scale_factor):
        batch_size, num_patches, channels = latents.shape

        # VAE applies 8x compression on images but we must also account for packing which requires
        # latent height and width to be divisible by 2.
        height = 2 * (int(height) // (vae_scale_factor * 2))
        width = 2 * (int(width) // (vae_scale_factor * 2))

        latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)

        latents = latents.reshape(batch_size, channels // (2 * 2), height, width)

        return latents

    @staticmethod
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

    def vae_encode(self, *args, **kwargs):
        """VAE encode"""
        return self.main_engine.vae_encode(*args, **kwargs)

    def vae_decode(self, *args, **kwargs):
        """VAE decode"""
        return self.main_engine.vae_decode(*args, **kwargs)

    def denoise(self, *args, **kwargs):
        """Denoise function"""
        return self.main_engine.denoise(*args, **kwargs)

    def _tensor_to_frame(self, *args, **kwargs):
        """Convert torch.tensor to PIL image"""
        return self.main_engine._tensor_to_frame(*args, **kwargs)

    def encode_image(self, image):
        image_encoder = self.main_engine["helpers"]["image_encoder"]
        return image_encoder(image)

    def prepare_ip_adapter_image_embeds(
        self, ip_adapter_image, ip_adapter_image_embeds, device, num_images
    ):
        if not self.transformer:
            self.load_component_by_type("transformer")

        image_embeds = []
        if ip_adapter_image_embeds is None:
            if not isinstance(ip_adapter_image, list):
                ip_adapter_image = [ip_adapter_image]

            if (
                len(ip_adapter_image)
                != self.transformer.encoder_hid_proj.num_ip_adapters
            ):
                raise ValueError(
                    f"`ip_adapter_image` must have same length as the number of IP Adapters. Got {len(ip_adapter_image)} images and {self.transformer.encoder_hid_proj.num_ip_adapters} IP Adapters."
                )

            for single_ip_adapter_image in ip_adapter_image:
                single_image_embeds = self.encode_image(single_ip_adapter_image)
                image_embeds.append(single_image_embeds[None, :])
        else:
            if not isinstance(ip_adapter_image_embeds, list):
                ip_adapter_image_embeds = [ip_adapter_image_embeds]

            if (
                len(ip_adapter_image_embeds)
                != self.transformer.encoder_hid_proj.num_ip_adapters
            ):
                raise ValueError(
                    f"`ip_adapter_image_embeds` must have same length as the number of IP Adapters. Got {len(ip_adapter_image_embeds)} image embeds and {self.transformer.encoder_hid_proj.num_ip_adapters} IP Adapters."
                )

            for single_image_embeds in ip_adapter_image_embeds:
                image_embeds.append(single_image_embeds)

        ip_adapter_image_embeds = []
        for single_image_embeds in image_embeds:
            single_image_embeds = torch.cat([single_image_embeds] * num_images, dim=0)
            single_image_embeds = single_image_embeds.to(device=device)
            ip_adapter_image_embeds.append(single_image_embeds)

        return ip_adapter_image_embeds

    def _get_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
        image=None,
        offload=True,
        timestep=None,
    ):
        
        if image is not None:
            image_latents = self.vae_encode(image, offload=offload)
            image_latent_height, image_latent_width = image_latents.shape[2:]
            if timestep is None:
                image_latents = self._pack_latents(
                    image_latents, batch_size, num_channels_latents, image_latent_height, image_latent_width
                )
                latent_image_ids = self._prepare_latent_image_ids(
                    batch_size, image_latent_height // 2, image_latent_width // 2, device, dtype
                )
                # image ids are the same as latent ids with the first dimension set to 1 instead of 0
                latent_image_ids[..., 0] = 1
            else:
                latent_image_ids = None
        else:
            image_latents = None
            latent_image_ids = None
            
        # VAE applies 8x compression on images but we must also account for packing which requires
        # latent height and width to be divisible by 2.
        height = 2 * (int(height) // (self.vae_scale_factor * 2))
        width = 2 * (int(width) // (self.vae_scale_factor * 2))

        shape = (batch_size, num_channels_latents, height, width)

        if latents is not None:
            latent_ids = self._prepare_latent_image_ids(
                batch_size, height // 2, width // 2, device, dtype
            )
            return latents.to(device=device, dtype=dtype), latent_ids
    
        noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            
        if timestep is not None:
            assert image_latents is not None, "Image latents are required for timestep scaling"
            image_latents = torch.cat([image_latents] * batch_size, dim=0)
            latents = self.scheduler.scale_noise(image_latents, timestep, noise)
        else:
            latents = noise

        latents = self._pack_latents(
            latents, batch_size, num_channels_latents, height, width
        )
        
        latent_ids = self._prepare_latent_image_ids(
            batch_size, height // 2, width // 2, device, dtype
        )
        
        if image_latents is not None and timestep is None:
            return latents, image_latents, latent_ids, latent_image_ids

        return latents, latent_ids

    @staticmethod
    def _prepare_latent_image_ids(batch_size, height, width, device, dtype):
        latent_image_ids = torch.zeros(height, width, 3)
        latent_image_ids[..., 1] = (
            latent_image_ids[..., 1] + torch.arange(height)[:, None]
        )
        latent_image_ids[..., 2] = (
            latent_image_ids[..., 2] + torch.arange(width)[None, :]
        )

        latent_image_id_height, latent_image_id_width, latent_image_id_channels = (
            latent_image_ids.shape
        )

        latent_image_ids = latent_image_ids.reshape(
            latent_image_id_height * latent_image_id_width, latent_image_id_channels
        )

        return latent_image_ids.to(device=device, dtype=dtype)
    
    def encode_prompt(
        self,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        prompt_2: Union[str, List[str]] = None,
        negative_prompt_2: Union[str, List[str]] = None,
        use_cfg_guidance: bool = True,
        offload: bool = True,
        num_images: int = 1,
        text_encoder_kwargs: Optional[Dict[str, Any]] = {},
        text_encoder_2_kwargs: Optional[Dict[str, Any]] = {},
    ):
        if not hasattr(self, "text_encoder") or not self.text_encoder:
            self.load_component_by_name("text_encoder")

        self.to_device(self.text_encoder)

        pooled_prompt_embeds = self.text_encoder.encode(
            f"{prompt}",
            device=self.device,
            num_videos_per_prompt=num_images,
            output_type="pooler_output",
            **text_encoder_kwargs,
        )

        if negative_prompt is not None and use_cfg_guidance:
            negative_pooled_prompt_embeds = self.text_encoder.encode(
                negative_prompt,
                device=self.device,
                num_videos_per_prompt=num_images,
                output_type="pooler_output",
                **text_encoder_kwargs,
            )
        else:
            negative_pooled_prompt_embeds = None

        if offload:
            self._offload(self.text_encoder)

        if not hasattr(self, "text_encoder_2") or not self.text_encoder_2:
            self.load_component_by_name("text_encoder_2")

        self.to_device(self.text_encoder_2)

        if not prompt_2:
            prompt_2 = prompt

        if not negative_prompt_2:
            negative_prompt_2 = negative_prompt

        prompt_embeds = self.text_encoder_2.encode(
            prompt_2,
            device=self.device,
            num_videos_per_prompt=num_images,
            output_type="hidden_states",
            **text_encoder_2_kwargs,
        )

        text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(
            device=self.device, dtype=prompt_embeds.dtype
        )

        if negative_prompt_2 is not None and use_cfg_guidance:
            negative_prompt_embeds = self.text_encoder_2.encode(
                negative_prompt_2,
                device=self.device,
                num_videos_per_prompt=num_images,
                output_type="hidden_states",
                **text_encoder_2_kwargs,
            )
            negative_text_ids = torch.zeros(negative_prompt_embeds.shape[1], 3).to(
                device=self.device, dtype=negative_prompt_embeds.dtype
            )
        else:
            negative_prompt_embeds = None
            negative_text_ids = None
            
        return pooled_prompt_embeds, negative_pooled_prompt_embeds, prompt_embeds, negative_prompt_embeds, text_ids, negative_text_ids
    
    
    def resize_to_preferred_resolution(self, image: Image.Image):
        PREFERRED_KONTEXT_RESOLUTIONS = [
            (672, 1568),
            (688, 1504),
            (720, 1456),
            (752, 1392),
            (800, 1328),
            (832, 1248),
            (880, 1184),
            (944, 1104),
            (1024, 1024),
            (1104, 944),
            (1184, 880),
            (1248, 832),
            (1328, 800),
            (1392, 752),
            (1456, 720),
            (1504, 688),
            (1568, 672),
        ]
        
        original_width, original_height = image.size
        original_aspect = original_width / original_height
        
        best_resolution = None
        min_area_diff = float('inf')
        
        for width, height in PREFERRED_KONTEXT_RESOLUTIONS:
            target_aspect = width / height
            area_diff = abs((width * height) - (original_width * original_height))
            aspect_diff = abs(target_aspect - original_aspect)
            
            if area_diff < min_area_diff and aspect_diff < 0.2:
                min_area_diff = area_diff
                best_resolution = (width, height)
        
        if best_resolution is None:
            best_resolution = min(PREFERRED_KONTEXT_RESOLUTIONS, 
                                key=lambda res: abs((res[0] * res[1]) - (original_width * original_height)))
        
        return image.resize(best_resolution, Image.Resampling.LANCZOS)
    
    def prepare_mask_latents(
        self,
        mask,
        masked_image,
        batch_size,
        num_channels_latents,
        num_images_per_prompt,
        height,
        width,
        dtype,
        device,
        offload=False,
    ):
        # 1. calculate the height and width of the latents
        # VAE applies 8x compression on images but we must also account for packing which requires
        # latent height and width to be divisible by 2.
        height = 2 * (int(height) // (self.vae_scale_factor * 2))
        width = 2 * (int(width) // (self.vae_scale_factor * 2))

        # 2. encode the masked image
        if masked_image.shape[1] == num_channels_latents:
            masked_image_latents = masked_image
        else:
            masked_image_latents = self.vae_encode(masked_image, offload=offload)

        # 3. duplicate mask and masked_image_latents for each generation per prompt, using mps friendly method
        batch_size = batch_size * num_images_per_prompt
        if mask.shape[0] < batch_size:
            if not batch_size % mask.shape[0] == 0:
                raise ValueError(
                    "The passed mask and the required batch size don't match. Masks are supposed to be duplicated to"
                    f" a total batch size of {batch_size}, but {mask.shape[0]} masks were passed. Make sure the number"
                    " of masks that you pass is divisible by the total requested batch size."
                )
            mask = mask.repeat(batch_size // mask.shape[0], 1, 1, 1)
        if masked_image_latents.shape[0] < batch_size:
            if not batch_size % masked_image_latents.shape[0] == 0:
                raise ValueError(
                    "The passed images and the required batch size don't match. Images are supposed to be duplicated"
                    f" to a total batch size of {batch_size}, but {masked_image_latents.shape[0]} images were passed."
                    " Make sure the number of images that you pass is divisible by the total requested batch size."
                )
            masked_image_latents = masked_image_latents.repeat(batch_size // masked_image_latents.shape[0], 1, 1, 1)

        # 4. pack the masked_image_latents
        # batch_size, num_channels_latents, height, width -> batch_size, height//2 * width//2 , num_channels_latents*4
        masked_image_latents = self._pack_latents(
            masked_image_latents,
            batch_size,
            num_channels_latents,
            height,
            width,
        )

        # 5.resize mask to latents shape we we concatenate the mask to the latents
        mask = mask[:, 0, :, :]  # batch_size, 8 * height, 8 * width (mask has not been 8x compressed)
        mask = mask.view(
            batch_size, height, self.vae_scale_factor, width, self.vae_scale_factor
        )  # batch_size, height, 8, width, 8
        mask = mask.permute(0, 2, 4, 1, 3)  # batch_size, 8, 8, height, width
        mask = mask.reshape(
            batch_size, self.vae_scale_factor * self.vae_scale_factor, height, width
        )  # batch_size, 8*8, height, width

        # 6. pack the mask:
        # batch_size, 64, height, width -> batch_size, height//2 * width//2 , 64*2*2
        mask = self._pack_latents(
            mask,
            batch_size,
            self.vae_scale_factor * self.vae_scale_factor,
            height,
            width,
        )
        mask = mask.to(device=device, dtype=dtype)

        return mask, masked_image_latents
