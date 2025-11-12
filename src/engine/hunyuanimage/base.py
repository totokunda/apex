import torch
from diffusers.utils.torch_utils import randn_tensor
from typing import Union, List, Optional, Dict, Any, TYPE_CHECKING, Callable
from transformers import ByT5Tokenizer, Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer, T5EncoderModel
from diffusers.image_processor import VaeImageProcessor
# Typing-only linkage to BaseEngine for IDE navigation and autocompletion,
# while avoiding a runtime dependency/import cycle.
if TYPE_CHECKING:
    from src.engine.base_engine import BaseEngine  # noqa: F401
    BaseClass = BaseEngine  # type: ignore
else:
    BaseClass = object
    
import re

class HunyuanImageBaseEngine(BaseClass):
    """Base class for HunyuanImage engine implementations containing common functionality"""

    def __init__(self, main_engine):
        self.main_engine = main_engine
        self.device = main_engine.device
        self.logger = main_engine.logger
        self.vae_scale_factor = main_engine.vae_scale_factor
        
        self.tokenizer_max_length = 1000
        self.tokenizer_2_max_length = 128
        self.prompt_template_encode = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>"
        self.prompt_template_encode_start_idx = 34
        self.default_sample_size = 64
    
    def __getattr__(self, name: str):  # noqa: D401
        """Delegate attribute access to the composed BaseEngine when not found here."""
        try:
            return getattr(self.main_engine, name)
        except AttributeError as exc:
            raise AttributeError(f"{self.__class__.__name__!s} has no attribute '{name}'") from exc

    # Improve editor introspection (e.g., autocomplete) by exposing attributes of main_engine
    def __dir__(self):
        return sorted(set[str](list[str](super().__dir__()) + dir(self.main_engine)))
        
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
    def helpers(self):
        return self.main_engine.helpers 
    
    def _get_qwen_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        tokenizer_max_length: int = 1000,
        template: str = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>",
        drop_idx: int = 34,
        hidden_state_skip_layer: int = 2,
    ):
        device = device or self.device
        dtype = dtype or self.component_dtypes["text_encoder"]

        prompt = [prompt] if isinstance(prompt, str) else prompt

        txt = [template.format(e) for e in prompt]

        if self.text_encoder is None:
            self.load_component_by_name("text_encoder")

        encoder_hidden_states, attention_mask = self.text_encoder.encode(
            txt,
            max_sequence_length=tokenizer_max_length + drop_idx,
            pad_to_max_length=True,
            use_attention_mask=True,
            return_attention_mask=True,
            clean_text=False,
            output_type="raw",
        )
        

        prompt_embeds = encoder_hidden_states.hidden_states[-(hidden_state_skip_layer + 1)]
        prompt_embeds = prompt_embeds[:, drop_idx:]
        encoder_attention_mask = attention_mask[:, drop_idx:]

        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        encoder_attention_mask = encoder_attention_mask.to(device=device)

        return prompt_embeds, encoder_attention_mask

    def _get_byt5_prompt_embeds(
        self,
        prompt: str,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        tokenizer_max_length: int = 128,
    ):
        device = device or self.device
        dtype = dtype or self.component_dtypes["text_encoder"]

        if isinstance(prompt, list):
            raise ValueError("byt5 prompt should be a string")
        elif prompt is None:
            raise ValueError("byt5 prompt should not be None")
        
        if not hasattr(self, "text_encoder_2") or not self.text_encoder_2:
            self.load_component_by_name("text_encoder_2")
            
    
        prompt_embeds, attention_mask = self.text_encoder_2.encode(
            prompt,
            max_sequence_length=tokenizer_max_length,
            pad_to_max_length=True,
            use_attention_mask=True,
            return_attention_mask=True,
            clean_text=False,
            output_type="raw",
        )
        
        prompt_embeds = prompt_embeds[0]

        # txt_tokens = tokenizer(
        #     prompt,
        #     padding="max_length",
        #     max_length=tokenizer_max_length,
        #     truncation=True,
        #     add_special_tokens=True,
        #     return_tensors="pt",
        # ).to(device)
# 
        # prompt_embeds = text_encoder(
        #     input_ids=txt_tokens.input_ids,
        #     attention_mask=txt_tokens.attention_mask.float(),
        # )[0]
# 
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        encoder_attention_mask = attention_mask.to(device=device)

        return prompt_embeds, encoder_attention_mask

    def encode_prompt(
        self,
        prompt: Union[str, List[str], None],
        batch_size: int = 1,
        num_images_per_prompt: int = 1,
        prompt_embeds: Optional[torch.Tensor] = None,
        prompt_embeds_mask: Optional[torch.Tensor] = None,
        prompt_embeds_2: Optional[torch.Tensor] = None,
        prompt_embeds_mask_2: Optional[torch.Tensor] = None,
        offload: bool = False,
    ):
        r"""

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            batch_size (`int`):
                batch size of prompts, defaults to 1
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. If not provided, text embeddings will be generated from `prompt` input
                argument.
            prompt_embeds_mask (`torch.Tensor`, *optional*):
                Pre-generated text mask. If not provided, text mask will be generated from `prompt` input argument.
            prompt_embeds_2 (`torch.Tensor`, *optional*):
                Pre-generated glyph text embeddings from ByT5. If not provided, will be generated from `prompt` input
                argument using self.tokenizer_2 and self.text_encoder_2.
            prompt_embeds_mask_2 (`torch.Tensor`, *optional*):
                Pre-generated glyph text mask from ByT5. If not provided, will be generated from `prompt` input
                argument using self.tokenizer_2 and self.text_encoder_2.
        """

        if prompt is None:
            prompt = [""] * batch_size

        prompt = [prompt] if isinstance(prompt, str) else prompt

        if prompt_embeds is None:
            prompt_embeds, prompt_embeds_mask = self._get_qwen_prompt_embeds(
                prompt=prompt,
                tokenizer_max_length=self.tokenizer_max_length,
                template=self.prompt_template_encode,
                drop_idx=self.prompt_template_encode_start_idx,
            )
            
        if offload:
            self._offload(self.text_encoder)

        if prompt_embeds_2 is None:
            prompt_embeds_2_list = []
            prompt_embeds_mask_2_list = []

            glyph_texts = [self.extract_glyph_text(p) for p in prompt]
            text_encoder_config = self.load_config_by_name("text_encoder_2")
            for glyph_text in glyph_texts:
                if glyph_text is None:
                    glyph_text_embeds = torch.zeros(
                        (1, self.tokenizer_2_max_length, text_encoder_config['d_model']), device=self.device
                    )
                    glyph_text_embeds_mask = torch.zeros(
                        (1, self.tokenizer_2_max_length), device=self.device, dtype=torch.int64
                    )
                else:
                    glyph_text_embeds, glyph_text_embeds_mask = self._get_byt5_prompt_embeds(
                        prompt=glyph_text,
                        device=self.device,
                        tokenizer_max_length=self.tokenizer_2_max_length,
                    )

                prompt_embeds_2_list.append(glyph_text_embeds)
                prompt_embeds_mask_2_list.append(glyph_text_embeds_mask)
                
            
            if offload and hasattr(self, "text_encoder_2"):
                self._offload(self.text_encoder_2)

            prompt_embeds_2 = torch.cat(prompt_embeds_2_list, dim=0)
            prompt_embeds_mask_2 = torch.cat(prompt_embeds_mask_2_list, dim=0)

        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)
        prompt_embeds_mask = prompt_embeds_mask.repeat(1, num_images_per_prompt, 1)
        prompt_embeds_mask = prompt_embeds_mask.view(batch_size * num_images_per_prompt, seq_len)

        _, seq_len_2, _ = prompt_embeds_2.shape
        prompt_embeds_2 = prompt_embeds_2.repeat(1, num_images_per_prompt, 1)
        prompt_embeds_2 = prompt_embeds_2.view(batch_size * num_images_per_prompt, seq_len_2, -1)
        prompt_embeds_mask_2 = prompt_embeds_mask_2.repeat(1, num_images_per_prompt, 1)
        prompt_embeds_mask_2 = prompt_embeds_mask_2.view(batch_size * num_images_per_prompt, seq_len_2)

        return prompt_embeds, prompt_embeds_mask, prompt_embeds_2, prompt_embeds_mask_2
    

    @staticmethod
    def extract_glyph_text(prompt: str):
        """
        Extract text enclosed in quotes for glyph rendering.

        Finds text in single quotes, double quotes, and Chinese quotes, then formats it for byT5 processing.

        Args:
            prompt: Input text prompt

        Returns:
            Formatted glyph text string or None if no quoted text found
        """
        text_prompt_texts = []
        pattern_quote_single = r"\'(.*?)\'"
        pattern_quote_double = r"\"(.*?)\""
        pattern_quote_chinese_single = r"‘(.*?)’"
        pattern_quote_chinese_double = r"“(.*?)”"

        matches_quote_single = re.findall(pattern_quote_single, prompt)
        matches_quote_double = re.findall(pattern_quote_double, prompt)
        matches_quote_chinese_single = re.findall(pattern_quote_chinese_single, prompt)
        matches_quote_chinese_double = re.findall(pattern_quote_chinese_double, prompt)

        text_prompt_texts.extend(matches_quote_single)
        text_prompt_texts.extend(matches_quote_double)
        text_prompt_texts.extend(matches_quote_chinese_single)
        text_prompt_texts.extend(matches_quote_chinese_double)

        if text_prompt_texts:
            glyph_text_formatted = ". ".join([f'Text "{text}"' for text in text_prompt_texts]) + ". "
        else:
            glyph_text_formatted = None

        return glyph_text_formatted
    

    def get_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        seed,
        generator,
        latents=None,
    ):
        if seed is not None:
            generator = torch.Generator(device=device).manual_seed(seed)
            
        height = int(height) // self.vae_scale_factor
        width = int(width) // self.vae_scale_factor

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
    
    def _render_step(self, latents: torch.Tensor, render_on_step_callback: Callable):
        """Decode latents and render a preview image during denoising."""
        image = self.vae_decode(latents, offload=True)
        image = self._tensor_to_frame(image)
        render_on_step_callback(image[0])