import hashlib
import os
import re
import time
from typing import Dict, List, Tuple

import torch
from safetensors.torch import safe_open
from safetensors.torch import save_file
from src.utils.defaults import DEFAULT_CACHE_PATH


class CacheMixin:
    enable_cache: bool = True
    cache_file: str | None = None
    max_cache_size: int | None = None
    model_path: str | None = None

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
