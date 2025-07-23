import transformers
from typing import Dict, Any, List, Literal
import torch
from src.text_encoder.tokenizer import fetch_and_save_tokenizer_from_config
from types import ModuleType
from src.utils.module_utils import find_class_recursive
from accelerate import init_empty_weights
import os
from safetensors.torch import load_model
import ftfy
import re
import html


class TextEncoder(torch.nn.Module):
    def __init__(
        self, config: Dict[str, Any], no_weights: bool = False, *args, **kwargs
    ):
        super().__init__()
        self.base = config.get("base")
        # check if base has transformers in it if so remove it
        if "transformer_models" in self.base:
            self.base = self.base.replace("transformer_models.", "")
        self.model_path = config.get("model_path")
        self.config_path = config.get("config_path")
        self.config = config.get("config", {})
        self.tokenizer = fetch_and_save_tokenizer_from_config(
            self.model_path,
            self.config_path,
            self.config,
            tokenizer_class=config.get("tokenizer_class", None),
            tokenizer_name=config.get("tokenizer_name", None),
            **config.get("tokenizer_kwargs", {}),
        )
        self.model = self._load_model(no_weights)

    def basic_clean(self, text):
        text = ftfy.fix_text(text)
        text = html.unescape(html.unescape(text))
        return text.strip()

    def whitespace_clean(self, text):
        text = re.sub(r"\s+", " ", text)
        text = text.strip()
        return text

    def prompt_clean(self, text):
        text = self.whitespace_clean(self.basic_clean(text))
        return text

    def _load_model(self, no_weights: bool = False):
        # get model class from recursively search transformers
        model_class = find_class_recursive(transformers, self.base)
        if model_class is None:
            raise ValueError(f"Model class {self.base} not found in transformers")

        if "torch_dtype" in self.config:
            self.config["torch_dtype"] = getattr(torch, self.config["torch_dtype"])

        if self.model_path and os.path.isdir(self.model_path):
            if no_weights:
                with init_empty_weights():
                    model = model_class.from_pretrained(self.model_path, **self.config)
            else:
                model = model_class.from_pretrained(self.model_path, **self.config)
        else:
            with init_empty_weights():
                model = model_class(**self.config)

            if no_weights:
                return model

            if self.model_path:
                if not os.path.exists(self.model_path):
                    raise FileNotFoundError(
                        f"Model file not found at {self.model_path}"
                    )

                if self.model_path.endswith(".safetensors"):
                    load_model(model, self.model_path)
                else:
                    state_dict = torch.load(
                        self.model_path,
                        map_location="cpu",
                        weights_only=True,
                        mmap=True,
                    )
                    model.load_state_dict(state_dict, strict=True)
        return model

    @torch.inference_mode()
    def encode(
        self,
        text: str | List[str],
        max_sequence_length: int = 512,
        pad_to_max_length: bool = True,
        num_videos_per_prompt: int = 1,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
        batch_size: int = 1,
        return_attention_mask: bool = False,
        use_mask_in_input: bool = True,
        use_position_ids: bool = False,
        pad_with_zero: bool = True,
        clean_text: bool = True,
        output_type: Literal["hidden_states", "pooler_output"] = "hidden_states",
    ):
        if isinstance(text, str):
            text = [text]
        if clean_text:
            text = [self.prompt_clean(t) for t in text]

        text_inputs = self.tokenizer(
            text,
            padding="max_length" if pad_to_max_length else "longest",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask
        seq_lens = mask.gt(0).sum(dim=1).long()
        mask = mask.bool()

        inputs = {"input_ids": text_input_ids.to(device=self.model.device)}
        # check if model takes position ids as input
        if use_position_ids:
            position_ids = torch.arange(text_input_ids.shape[1]).expand(
                batch_size, text_input_ids.shape[1]
            )
            position_ids = position_ids.to(dtype=torch.long, device=self.model.device)
            inputs["position_ids"] = position_ids
        if use_mask_in_input:
            inputs["attention_mask"] = mask.to(device=self.model.device)

        result = self.model(
            **inputs, output_hidden_states=output_type == "hidden_states"
        )

        if output_type == "hidden_states":
            prompt_embeds = result.last_hidden_state
        elif output_type == "pooler_output":
            prompt_embeds = result.pooler_output
        else:
            raise ValueError(f"Invalid output type: {output_type}")

        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        if output_type == "pooler_output":
            prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt)
            prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, -1)
        elif pad_with_zero and output_type == "hidden_states":
            prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
            prompt_embeds = torch.stack(
                [
                    (
                        torch.cat(
                            [u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))]
                        )
                        if pad_to_max_length
                        else u
                    )
                    for u in prompt_embeds
                ],
                dim=0,
            )
            # duplicate text embeddings for each generation per prompt, using mps friendly method
            _, seq_len, _ = prompt_embeds.shape
            prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
            mask = mask.repeat(1, num_videos_per_prompt)

            prompt_embeds = prompt_embeds.view(
                batch_size * num_videos_per_prompt, seq_len, -1
            )
        else:
            _, seq_len, _ = prompt_embeds.shape
            prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)

            prompt_embeds = prompt_embeds.view(
                batch_size * num_videos_per_prompt, seq_len, -1
            )
            mask = mask.repeat(1, num_videos_per_prompt)

        if return_attention_mask:
            return prompt_embeds, mask
        else:
            return prompt_embeds
