import transformers
from typing import Dict, Any, List, Literal
import torch
from src.text_encoder.tokenizer import fetch_and_save_tokenizer_from_config
from src.mixins.loader_mixin import LoaderMixin
import ftfy
import re
import html


class TextEncoder(torch.nn.Module, LoaderMixin):
    def __init__(
        self, config: Dict[str, Any], no_weights: bool = False, *args, **kwargs
    ):
        super().__init__()
        self.base = config.get("base")
        # check if base has transformers in it if so remove it
        if "transformer" in self.base:
            self.base = self.base.replace("transformer.", "")
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
        self.model = self._load_model(
            {
                "config": self.config,
                "config_path": self.config_path,
                "model_path": self.model_path,
                "base": self.base,
            },
            module_name="transformers",
            no_weights=no_weights,
            key_map=config.get("key_map", {}),
            extra_kwargs=config.get("extra_kwargs", {}),
        )

    def basic_clean(self, text):
        text = ftfy.fix_text(text)
        text = html.unescape(html.unescape(text))
        return text.strip()

    def whitespace_clean(self, text):
        text = re.sub(r"\s+", " ", text)
        text = text.strip()
        return text

    def prompt_clean(self, text, lower_case: bool = False):
        text = self.whitespace_clean(self.basic_clean(text))
        if lower_case:
            text = text.lower()
        return text

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
        add_special_tokens: bool = True,
        return_attention_mask: bool = False,
        use_mask_in_input: bool = False,
        use_position_ids: bool = False,
        use_token_type_ids: bool = False,
        pad_with_zero: bool = True,
        clean_text: bool = True,
        output_type: Literal["hidden_states", "pooler_output"] = "hidden_states",
        lower_case: bool = False,
    ):
        if isinstance(text, str):
            text = [text]
        if clean_text:
            text = [self.prompt_clean(t, lower_case=lower_case) for t in text]

        text_inputs = self.tokenizer(
            text,
            padding="max_length" if pad_to_max_length else "longest",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=add_special_tokens,
            return_tensors="pt",
            return_attention_mask=True,
        )
  
        text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask
        seq_lens = mask.gt(0).sum(dim=1).long()
        #mask = mask.bool()

        inputs = {"input_ids": text_input_ids.to(device=self.model.device)}
        
        if use_position_ids:
            position_ids = torch.arange(text_input_ids.shape[1]).expand(
                batch_size, text_input_ids.shape[1]
            )
            position_ids = position_ids.to(dtype=torch.long, device=self.model.device)
            inputs["position_ids"] = position_ids
        
        if use_token_type_ids:
            inputs["token_type_ids"] = torch.zeros_like(text_input_ids).to(
                device=self.model.device
            )
        
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
