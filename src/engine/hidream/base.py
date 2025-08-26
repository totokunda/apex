import torch
from diffusers.utils.torch_utils import randn_tensor
from PIL import Image
import math

class HidreamBaseEngine:
    """Base class for Hidream engine implementations containing common functionality"""

    def __init__(self, main_engine):
        self.main_engine = main_engine
        self.device = main_engine.device
        self.logger = main_engine.logger
        self.vae_scale_factor = main_engine.vae_scale_factor
        self.num_channels_latents = main_engine.num_channels_latents
        self.image_processor = main_engine.image_processor
        self.default_sample_size = main_engine.default_sample_size

    @property
    def text_encoder(self):
        return self.main_engine.text_encoder

    @property
    def text_encoder_2(self):
        return getattr(self.main_engine, "text_encoder_2", None)

    @property
    def text_encoder_3(self):
        return getattr(self.main_engine, "text_encoder_3", None)
    
    @property
    def refiner(self):
        return getattr(self.main_engine, "refiner", None)
    
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
    
    @property
    def helpers(self):
        return self.main_engine.helpers
    
    def load_config_by_type(self, component_type: str):
        """Load a component by type"""
        return self.main_engine.load_config_by_type(component_type)

    def load_config_by_name(self, component_name: str):
        """Load a component by name"""
        return self.main_engine.load_config_by_name(component_name)

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

    def _get_latents(
        self,
        batch_size: int,
        num_channels_latents: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device,
        seed: int,
        generator=None,
        latents=None,
    ):
        # VAE applies 8x compression on images but we must also account for packing which requires
        # latent height and width to be divisible by 2.
        height = 2 * (int(height) // (self.vae_scale_factor * 2))
        width = 2 * (int(width) // (self.vae_scale_factor * 2))

        shape = (batch_size, num_channels_latents, height, width)

        if seed is not None:
            generator = torch.Generator(device=device).manual_seed(seed)

        if latents is not None:
            return latents.to(device=device, dtype=dtype)

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)

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
    ):
        # VAE applies 8x compression on images but we must also account for packing which requires
        # latent height and width to be divisible by 2.
        height = 2 * (int(height) // (self.vae_scale_factor * 2))
        width = 2 * (int(width) // (self.vae_scale_factor * 2))

        shape = (batch_size, num_channels_latents, height, width)

        if latents is not None:
            return latents.to(device=device, dtype=dtype)

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)

        return latents

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
        prompt,
        prompt_2,
        prompt_3,
        prompt_4,
        negative_prompt,
        negative_prompt_2,
        negative_prompt_3,
        negative_prompt_4,
        text_encoder_kwargs,
        text_encoder_2_kwargs,
        text_encoder_3_kwargs,
        num_images,
        use_cfg_guidance,
        offload,
    ):
        if not hasattr(self, "text_encoder") or not self.text_encoder:
            self.load_component_by_name("text_encoder")

        self.to_device(self.text_encoder)

        pooled_prompt_embeds_1 = self.text_encoder.encode(
            f"<|startoftext|>{prompt}",
            device=self.device,
            num_videos_per_prompt=num_images,
            output_type="text_embeds",
            **text_encoder_kwargs,
        )
        
        if use_cfg_guidance and negative_prompt is None:
            negative_prompt = ""

        if negative_prompt is not None and use_cfg_guidance:
            negative_pooled_prompt_embeds_1 = self.text_encoder.encode(
                f"<|startoftext|>{negative_prompt}",
                device=self.device,
                num_videos_per_prompt=num_images,
                output_type="text_embeds",
                **text_encoder_kwargs,
            )
        else:
            negative_pooled_prompt_embeds_1 = None

        if offload:
            self._offload(self.text_encoder)

        if not hasattr(self, "text_encoder_2") or not self.text_encoder_2:
            self.load_component_by_name("text_encoder_2")

        self.to_device(self.text_encoder_2)

        if not prompt_2:
            prompt_2 = prompt

        if not negative_prompt_2:
            negative_prompt_2 = negative_prompt
            
        pooled_prompt_embeds_2 = self.text_encoder_2.encode(
            f"<|startoftext|>{prompt_2}<|endoftext|>",
            device=self.device,
            num_videos_per_prompt=num_images,
            output_type="text_embeds",
            **text_encoder_2_kwargs,
        )

        if negative_prompt_2 is not None and use_cfg_guidance:
            negative_pooled_prompt_embeds_2 = self.text_encoder_2.encode(
                f"<|startoftext|>{negative_prompt_2}<|endoftext|>",
                device=self.device,
                num_videos_per_prompt=num_images,
                output_type="text_embeds",
                **text_encoder_2_kwargs,
            )
        else:
            negative_pooled_prompt_embeds_2 = None

        if offload:
            self._offload(self.text_encoder_2)

        if not hasattr(self, "text_encoder_3") or not self.text_encoder_3:
            self.load_component_by_name("text_encoder_3")

        self.to_device(self.text_encoder_3)

        if not prompt_3:
            prompt_3 = prompt

        if not negative_prompt_3:
            negative_prompt_3 = negative_prompt

        prompt_embeds = self.text_encoder_3.encode(
            prompt_3,
            device=self.device,
            num_videos_per_prompt=num_images,
            **text_encoder_3_kwargs,
        )

        if negative_prompt_3 is not None and use_cfg_guidance:
            negative_prompt_embeds = self.text_encoder_3.encode(
                negative_prompt_3,
                device=self.device,
                num_videos_per_prompt=num_images,
                **text_encoder_3_kwargs,
            )
        else:
            negative_prompt_embeds = None

        if offload:
            self._offload(self.text_encoder_3)
            
        pooled_prompt_embeds = torch.cat(
            [pooled_prompt_embeds_1, pooled_prompt_embeds_2], dim=-1
        ).view(num_images, -1)

        if use_cfg_guidance:
            negative_pooled_prompt_embeds = torch.cat(
                [negative_pooled_prompt_embeds_1, negative_pooled_prompt_embeds_2],
                dim=-1,
            ).view(num_images, -1)
            

        if not prompt_4:
            prompt_4 = prompt
        
        if not negative_prompt_4:
            negative_prompt_4 = negative_prompt
        
        llama_encoder = self.helpers["llama"]
        self.to_device(llama_encoder)
        
        llama_prompt_embeds = llama_encoder(prompt_4, device=self.device, dtype=prompt_embeds.dtype, num_images_per_prompt=num_images)
        
        if negative_prompt_4 is not None and use_cfg_guidance:
            llama_negative_prompt_embeds = llama_encoder(negative_prompt_4, device=self.device, dtype=prompt_embeds.dtype, num_images_per_prompt=num_images)
        else:
            llama_negative_prompt_embeds = None
            
        if offload:
            self._offload(llama_encoder)

        return (
            prompt_embeds,
            negative_prompt_embeds,
            llama_prompt_embeds,
            llama_negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        )

    def resize_image(self, pil_image: Image.Image, image_size: int = 1024) -> Image.Image:
        while min(*pil_image.size) >= 2 * image_size:
            pil_image = pil_image.resize(
                tuple(x // 2 for x in pil_image.size), resample=Image.BOX
            )

        m = 16
        width, height = pil_image.width, pil_image.height
        S_max = image_size * image_size
        scale = S_max / (width * height)
        scale = math.sqrt(scale)

        new_sizes = [
            (round(width * scale) // m * m, round(height * scale) // m * m),
            (round(width * scale) // m * m, math.floor(height * scale) // m * m),
            (math.floor(width * scale) // m * m, round(height * scale) // m * m),
            (math.floor(width * scale) // m * m, math.floor(height * scale) // m * m),
        ]
        new_sizes = sorted(new_sizes, key=lambda x: x[0] * x[1], reverse=True)

        for new_size in new_sizes:
            if new_size[0] * new_size[1] <= S_max:
                break

        s1 = width / new_size[0]
        s2 = height / new_size[1]
        if s1 < s2:
            pil_image = pil_image.resize([new_size[0], round(height / s1)], resample=Image.BICUBIC)
            top = (round(height / s1) - new_size[1]) // 2
            pil_image = pil_image.crop((0, top, new_size[0], top + new_size[1]))
        else:
            pil_image = pil_image.resize([round(width / s2), new_size[1]], resample=Image.BICUBIC)
            left = (round(width / s2) - new_size[0]) // 2
            pil_image = pil_image.crop((left, 0, left + new_size[0], new_size[1]))

        return pil_image