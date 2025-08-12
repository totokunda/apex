from typing import Dict, Any, List, Literal, Tuple
import torch
from src.text_encoder.tokenizer import fetch_and_save_tokenizer_from_config
from src.mixins.loader_mixin import LoaderMixin
import ftfy
import re
import html
from src.utils.defaults import DEFAULT_CACHE_PATH, DEFAULT_COMPONENTS_PATH
import os
from safetensors import safe_open
from safetensors.torch import save_file
import hashlib
import time
from transformers import AutoTokenizer
from src.utils.module import find_class_recursive
import transformers

class TextEncoder(torch.nn.Module, LoaderMixin):
    def __init__(
        self,
        config: Dict[str, Any],
        no_weights: bool = False,
        enable_cache: bool = True,
        cache_file: str = None,
        max_cache_size: int = 100,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.base = config.get("base")
        # check if base has transformers in it if so remove it
        if "transformer" in self.base:
            self.base = self.base.replace("transformer.", "")
        self.model_path = config.get("model_path")
        self.config_path = config.get("config_path", None)
        self.tokenizer_path = config.get("tokenizer_path", None)
        self.config = config.get("config", {})
        self.enable_cache = enable_cache
        self.cache_file = cache_file
        self.max_cache_size = max_cache_size
        if self.enable_cache and self.cache_file is None:
            self.cache_file = os.path.join(
                DEFAULT_CACHE_PATH,
                f"text_encoder_{self.model_path.replace('/', '_')}.safetensors",
            )
        if self.tokenizer_path is not None:
            self.tokenizer_path = self._download(self.tokenizer_path, DEFAULT_COMPONENTS_PATH)
            tokenizer_class = find_class_recursive(transformers, config.get("tokenizer_class", "AutoTokenizer"))
            self.tokenizer = tokenizer_class.from_pretrained(self.tokenizer_path)
        else:
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
                "type": "text_encoder",
                "gguf_kwargs": config.get("gguf_kwargs", {}),
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

        prompt_hash = self.hash_prompt(text)

        if self.enable_cache:
            cached = self.load_cached_prompt(prompt_hash)
            if cached is not None:
                cached_embeds, cached_mask = cached

                # Move to requested dtype/device without altering mask dtype
                if dtype is not None:
                    cached_embeds = cached_embeds.to(dtype=dtype)
                if device is not None:
                    cached_embeds = cached_embeds.to(device=device)
                    if cached_mask is not None:
                        cached_mask = cached_mask.to(device=device)

                if return_attention_mask:
                    return cached_embeds, cached_mask
                else:
                    return cached_embeds

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
        # mask = mask.bool()

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

        if self.enable_cache:
            self.cache_prompt(
                prompt_hash,
                prompt_embeds,
                mask if return_attention_mask or True else None,
            )

        if return_attention_mask:
            return prompt_embeds, mask
        else:
            return prompt_embeds

    def hash_prompt(self, text: List[str]) -> str:
        hashes = [hashlib.sha256(str(t).encode()).hexdigest() for t in text]
        return ".".join(hashes)

    def get_cached_keys_for_prompt(
        self, prompt_hash: str
    ) -> Tuple[str | None, str | None]:
        """Return the most recent cache keys for given prompt hash.

        Returns a tuple: (prompt_embeds_key, attention_mask_key). Either may be None.
        Recognizes both timestamped and legacy formats.
        """
        if not self.enable_cache:
            return None, None

        if self.cache_file is None:
            self.cache_file = os.path.join(
                DEFAULT_CACHE_PATH,
                f"text_encoder_{self.model_path.replace('/', '_')}.safetensors",
            )

        if not os.path.exists(self.cache_file):
            return None, None

        key_pattern = re.compile(
            r"^(?:(?P<ts>\d{13,})_)?(?P<hash>[a-f0-9\.]+)_(?P<kind>prompt_embeds|attention_mask)$"
        )

        def parse_entry_key(key: str) -> Tuple[int, str, str] | None:
            match = key_pattern.match(key)
            if not match:
                return None
            ts_str = match.group("ts")
            ts = int(ts_str) if ts_str is not None else 0
            return ts, match.group("hash"), match.group("kind")

        latest_embed: Tuple[int, str] | None = None
        latest_mask: Tuple[int, str] | None = None

        try:
            with safe_open(self.cache_file, framework="pt", device="cpu") as f:
                for key in f.keys():
                    parsed = parse_entry_key(key)
                    if parsed is None:
                        continue
                    ts, hsh, kind = parsed
                    if hsh != prompt_hash:
                        continue
                    if kind == "prompt_embeds":
                        if latest_embed is None or ts >= latest_embed[0]:
                            latest_embed = (ts, key)
                    elif kind == "attention_mask":
                        if latest_mask is None or ts >= latest_mask[0]:
                            latest_mask = (ts, key)
        except Exception:
            return None, None

        return (
            latest_embed[1] if latest_embed is not None else None,
            latest_mask[1] if latest_mask is not None else None,
        )

    def load_cached_prompt(
        self, prompt_hash: str
    ) -> Tuple[torch.Tensor, torch.Tensor] | None:
        """Load cached tensors for the given prompt hash if present.

        Returns (prompt_embeds, attention_mask) on hit; otherwise None.
        """
        embed_key, mask_key = self.get_cached_keys_for_prompt(prompt_hash)
        if embed_key is None:
            return None
        try:
            with safe_open(self.cache_file, framework="pt", device="cpu") as f:
                prompt_embeds = f.get_tensor(embed_key)
                attention_mask = (
                    f.get_tensor(mask_key) if mask_key is not None else None
                )
            return prompt_embeds, attention_mask
        except Exception:
            return None

    def cache_prompt(
        self,
        prompt_hash: str,
        prompt_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> None:
        """Persist prompt embeddings and attention mask with LRU-style eviction.

        - Stores tensors in a single safetensors file under timestamped keys to track recency.
        - Removes any pre-existing entries for the same prompt hash (acts like an update/move-to-front).
        - If the number of unique cached prompts exceeds `max_cache_size`, evicts the oldest prompts first.

        Key format (new):
            "{timestamp_ms}_{prompt_hash}_prompt_embeds"
            "{timestamp_ms}_{prompt_hash}_attention_mask"

        Backward compatibility:
            Also recognizes legacy keys without timestamps like
            "{prompt_hash}_prompt_embeds" and "{prompt_hash}_attention_mask".
        """
        if not self.enable_cache:
            return

        # Ensure cache path exists
        if self.cache_file is None:
            self.cache_file = os.path.join(
                DEFAULT_CACHE_PATH,
                f"text_encoder_{self.model_path.replace('/', '_')}.safetensors",
            )
        os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)

        # Load existing cache tensors (best-effort)
        existing_tensors: Dict[str, torch.Tensor] = {}
        if os.path.exists(self.cache_file):
            try:
                with safe_open(self.cache_file, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        # Lazy: only load when needed for rewrite; we need all to rewrite cleanly
                        existing_tensors[key] = f.get_tensor(key)
            except Exception:
                # Corrupt or unreadable cache file; reset it
                existing_tensors = {}

        # Build index of entries by hash with their latest timestamp
        # Recognize both timestamped and legacy key formats
        key_pattern = re.compile(
            r"^(?:(?P<ts>\d{13,})_)?(?P<hash>[a-f0-9\.]+)_(?P<kind>prompt_embeds|attention_mask)$"
        )

        def parse_entry_key(key: str) -> Tuple[int, str, str] | None:
            match = key_pattern.match(key)
            if not match:
                return None
            ts_str = match.group("ts")
            ts = int(ts_str) if ts_str is not None else 0
            return ts, match.group("hash"), match.group("kind")

        entries_by_hash: Dict[str, Dict[str, Tuple[int, str]]] = {}
        # maps: prompt_hash -> { "prompt_embeds": (ts, key), "attention_mask": (ts, key) }
        for key in list(existing_tensors.keys()):
            parsed = parse_entry_key(key)
            if parsed is None:
                continue
            ts, hsh, kind = parsed
            kinds = entries_by_hash.setdefault(hsh, {})
            # Keep the most recent key per kind for that hash
            if kind not in kinds or ts >= kinds[kind][0]:
                kinds[kind] = (ts, key)

        # Remove previous entries for this prompt hash (any timestamp or legacy)
        if prompt_hash in entries_by_hash:
            for kind, (_ts, key) in list(entries_by_hash[prompt_hash].items()):
                existing_tensors.pop(key, None)
            entries_by_hash.pop(prompt_hash, None)

        # Insert the new entry with a fresh timestamp
        timestamp_ms = int(time.time() * 1000)
        embed_key = f"{timestamp_ms}_{prompt_hash}_prompt_embeds"
        mask_key = f"{timestamp_ms}_{prompt_hash}_attention_mask"
        existing_tensors[embed_key] = prompt_embeds.detach().to("cpu")
        existing_tensors[mask_key] = attention_mask.detach().to("cpu")

        # Update index with the new entries
        entries_by_hash[prompt_hash] = {
            "prompt_embeds": (timestamp_ms, embed_key),
            "attention_mask": (timestamp_ms, mask_key),
        }

        # Enforce max_cache_size (unique prompt hashes)
        unique_hashes = list(entries_by_hash.keys())
        num_prompts = len(unique_hashes)
        if (
            self.max_cache_size is not None
            and self.max_cache_size > 0
            and num_prompts > self.max_cache_size
        ):
            # Compute recency per hash (latest timestamp across kinds)
            hash_to_latest_ts = {
                hsh: max(ts_kind[0] for ts_kind in kinds.values())
                for hsh, kinds in entries_by_hash.items()
            }
            # Sort hashes by recency ascending (oldest first)
            eviction_order = sorted(hash_to_latest_ts.items(), key=lambda x: x[1])
            # Do not evict the newly added prompt; prioritize evicting others first
            eviction_candidates = [h for h, _ in eviction_order if h != prompt_hash]
            num_to_evict = num_prompts - self.max_cache_size
            for hsh in eviction_candidates[:num_to_evict]:
                kinds = entries_by_hash.pop(hsh, {})
                for _kind, (_ts, key) in kinds.items():
                    existing_tensors.pop(key, None)

        # Rewrite cache file atomically
        try:
            save_file(existing_tensors, self.cache_file)
        except Exception:
            # As a fallback, avoid crashing the caller; drop caching if write fails
            pass
