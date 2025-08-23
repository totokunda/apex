import torch
from diffusers.utils.torch_utils import randn_tensor
from typing import Union, List, Optional, Dict, Any

class QwenImageBaseEngine:
    """Base class for QwenImage engine implementations containing common functionality"""

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

        latents = latents.reshape(batch_size, channels // (2 * 2), 1, height, width)

        return latents

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

        shape = (batch_size, 1, num_channels_latents, height, width)
        
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
        latents = self._pack_latents(latents, batch_size, num_channels_latents, height, width)

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
    
    def encode_prompt(
        self,
        prompt: Union[str, List[str]] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        max_sequence_length: int = 1024,
        num_images_per_prompt: int = 1,
        text_encoder_kwargs: Optional[Dict[str, Any]] = {},
    ):

        prompt = [prompt] if isinstance(prompt, str) else prompt

        template = self.prompt_template_encode
        drop_idx = self.prompt_template_encode_start_idx
        txt = [template.format(e) for e in prompt]
        
        
        input_kwargs = {
            "text": txt,
            "max_sequence_length": self.tokenizer_max_length + drop_idx,
            "use_mask_in_input": True,
            "return_attention_mask": True,
            "output_type": "raw",
            **text_encoder_kwargs,
        }
        
        prompt_hash = self.text_encoder.hash_prompt(input_kwargs)
        cached = None
        if self.text_encoder.enable_cache:
            cached = self.text_encoder.load_cached_prompt(prompt_hash)
        
        if cached is not None:
            prompt_embeds, prompt_embeds_mask = cached
        else:
            encoder_hidden_states, attention_mask = self.text_encoder.encode(
                **input_kwargs,
            )
            hidden_states = encoder_hidden_states.hidden_states[-1]

            split_hidden_states = self._extract_masked_hidden(hidden_states, attention_mask)
            split_hidden_states = [e[drop_idx:] for e in split_hidden_states]
            attn_mask_list = [torch.ones(e.size(0), dtype=torch.long, device=e.device) for e in split_hidden_states]
            max_seq_len = max([e.size(0) for e in split_hidden_states])
            prompt_embeds = torch.stack(
                [torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))]) for u in split_hidden_states]
            )
            prompt_embeds_mask = torch.stack(
                [torch.cat([u, u.new_zeros(max_seq_len - u.size(0))]) for u in attn_mask_list]
            )

            prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

            prompt_embeds = prompt_embeds[:, :max_sequence_length]
            prompt_embeds_mask = prompt_embeds_mask[:, :max_sequence_length]

            _, seq_len, _ = prompt_embeds.shape
            prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
            prompt_embeds = prompt_embeds.view(num_images_per_prompt, seq_len, -1)
            prompt_embeds_mask = prompt_embeds_mask.repeat(1, num_images_per_prompt, 1)
            prompt_embeds_mask = prompt_embeds_mask.view(num_images_per_prompt, seq_len)

            if self.text_encoder.enable_cache:
                prompt_hash = self.text_encoder.hash_prompt(input_kwargs)

                self.text_encoder.cache_prompt(
                    prompt_hash,
                    prompt_embeds,
                    prompt_embeds_mask,
                )

        return prompt_embeds, prompt_embeds_mask
    
    def _extract_masked_hidden(self, hidden_states: torch.Tensor, mask: torch.Tensor):
        bool_mask = mask.bool()
        valid_lengths = bool_mask.sum(dim=1)
        selected = hidden_states[bool_mask]
        split_result = torch.split(selected, valid_lengths.tolist(), dim=0)

        return split_result