import torch
from diffusers.utils.torch_utils import randn_tensor
from typing import Union, List, Optional, Dict, Any

class ChromaBaseEngine:
    """Base class for Chroma engine implementations containing common functionality"""

    def __init__(self, main_engine):
        self.main_engine = main_engine
        self.device = main_engine.device
        self.logger = main_engine.logger
        self.vae_scale_factor = main_engine.vae_scale_factor
        self.num_channels_latents = main_engine.num_channels_latents
        self.image_processor = main_engine.image_processor
        self.tokenizer_max_length = 1024
        self.prompt_template_encode = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        self.prompt_template_encode_start_idx = 34
        self.default_sample_size = 128

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

    @staticmethod
    def _pack_latents(latents, batch_size, num_channels_latents, height, width):
        latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)

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

    def _get_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        seed,
        latents=None,
    ):
        # VAE applies 8x compression on images but we must also account for packing which requires
        # latent height and width to be divisible by 2.
        if seed is not None:
            generator = torch.Generator(device=device).manual_seed(seed)

        height = 2 * (int(height) // (self.vae_scale_factor * 2))
        width = 2 * (int(width) // (self.vae_scale_factor * 2))

        shape = (batch_size, num_channels_latents, height, width)

        if latents is not None:
            latent_image_ids = self._prepare_latent_image_ids(
                batch_size, height // 2, width // 2, device, dtype
            )
            return latents.to(device=device, dtype=dtype), latent_image_ids

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        latents = self._pack_latents(
            latents, batch_size, num_channels_latents, height, width
        )

        latent_image_ids = self._prepare_latent_image_ids(
            batch_size, height // 2, width // 2, device, dtype
        )

        return latents, latent_image_ids
    
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

    def _prepare_attention_mask(
        self,
        batch_size,
        sequence_length,
        dtype,
        attention_mask=None,
    ):
        if attention_mask is None:
            return attention_mask

        # Extend the prompt attention mask to account for image tokens in the final sequence
        attention_mask = torch.cat(
            [attention_mask, torch.ones(batch_size, sequence_length, device=attention_mask.device)],
            dim=1,
        )
        attention_mask = attention_mask.to(dtype)

        return attention_mask
    
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

    
    def encode_prompt(
        self,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        use_cfg_guidance: bool = True,
        offload: bool = True,
        num_images: int = 1,
        text_encoder_kwargs: Optional[Dict[str, Any]] = {},
    ):
        if not hasattr(self, "text_encoder") or not self.text_encoder:
            self.load_component_by_type("text_encoder")

        self.to_device(self.text_encoder)

        prompt_embeds, prompt_embeds_mask = self.text_encoder.encode(
            prompt,
            device=self.device,
            num_videos_per_prompt=num_images,
            return_attention_mask=True,
            output_type="hidden_states",
            **text_encoder_kwargs,
        )

        text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(
            device=self.device, dtype=prompt_embeds.dtype
        )

        if negative_prompt is not None and use_cfg_guidance:
            negative_prompt_embeds, negative_prompt_embeds_mask = self.text_encoder.encode(
                negative_prompt,
                device=self.device,
                num_videos_per_prompt=num_images,
                return_attention_mask=True,
                output_type="hidden_states",
                **text_encoder_kwargs,
            )
            
            negative_text_ids = torch.zeros(negative_prompt_embeds.shape[1], 3).to(
                device=self.device, dtype=negative_prompt_embeds.dtype
            )
            
        else:
            negative_prompt_embeds = None
            negative_text_ids = None
            
        if offload:
            self._offload(self.text_encoder)
            
        return prompt_embeds, prompt_embeds_mask, negative_prompt_embeds, negative_prompt_embeds_mask, text_ids, negative_text_ids