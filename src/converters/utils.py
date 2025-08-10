import json
import os
from typing import Dict, Any, Optional
from accelerate import init_empty_weights
from pydash import includes
from src.utils.module import find_class_recursive
import importlib
import re
from collections import Counter
import torch

TRANSFORMER_CONFIG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "transformer_configs"
)

VAE_CONFIG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "vae_configs"
)


def get_transformer_config(model_tag: str, config_path: str | None = None):
    if config_path is None:
        if "/" in model_tag:
            model_base, model_tag = model_tag.split("/")
        else:
            model_base = model_tag.split("_")[0]
        config_path = os.path.join(
            TRANSFORMER_CONFIG_DIR, model_base, f"{model_tag}.json"
        )
    with open(config_path, "r") as f:
        config = json.load(f)
    return config


def get_vae_config(vae_tag: str, config_path: str | None = None):
    if config_path is None:
        model_base = vae_tag.split("_")[0]
        config_path = os.path.join(VAE_CONFIG_DIR, model_base, f"{vae_tag}.json")

    with open(config_path, "r") as f:
        config = json.load(f)
    return config


def get_model_class(
    model_base: str, model_config: dict, model_type: str = "transformer"
):
    module_type = importlib.import_module(f"src.{model_type}.{model_base}.model")
    model_class = find_class_recursive(module_type, model_config["_class_name"])
    return model_class


def get_empty_model(model_class, config: dict):
    with init_empty_weights():
        model = model_class(**config)
    return model


def swap_scale_shift(t: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """[shift, scale]  ->  [scale, shift] along `dim`."""
    shift, scale = t.chunk(2, dim=dim)
    return torch.cat([scale, shift], dim=dim)


def swap_proj_gate(t: torch.Tensor) -> torch.Tensor:
    """[proj, gate]  ->  [gate, proj] for Gated-GeLU/SiLU MLPs."""
    proj, gate = t.chunk(2, dim=0)
    return torch.cat([gate, proj], dim=0)


def update_state_dict_(sd: Dict[str, Any], old_key: str, new_key: str):
    """Pop `old_key` (if still present) and write it back under `new_key`."""
    if old_key in sd:
        sd[new_key] = sd.pop(old_key)


def strip_common_prefix(
    src_state: Dict[str, Any],
    ref_state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Return a *new* state-dict whose keys no longer have an extra prefix.

    Parameters
    ----------
    src_state : dict
        The state_dict that might contain an unwanted prefix.
    ref_state : dict, optional
        A “clean” state_dict whose keys represent the desired names.
        If given, the function tries to find the shortest prefix `p` such that
        `k[len(p):] in ref_state` for *many* keys in `src_state`.
        If omitted, the function simply removes a unanimous first token
        (everything before the first '.') if all keys share it.

    Examples
    --------
    >>> clean_sd = {"blocks.0.attn1.to_q.weight": torch.randn(1)}
    >>> dirty_sd = {"model.diffusion_model.blocks.0.attn1.to_q.weight": torch.randn(1)}
    >>> strip_common_prefix(dirty_sd, clean_sd).keys()
    dict_keys(['blocks.0.attn1.to_q.weight'])
    """
    if ref_state is None:
        # Heuristic: do *all* keys share the same first token?
        first_tokens = {k.split(".", 1)[0] for k in src_state.keys()}
        if len(first_tokens) == 1:  # unanimous
            prefix = next(iter(first_tokens)) + "."
        else:
            return src_state  # nothing to strip
    else:
        ref_keys = set(ref_state.keys())
        prefix_counter: Counter[str] = Counter()

        # Look for candidate prefixes
        for k in src_state.keys():
            # Skip keys that already match
            if k in ref_keys:
                continue
            # Try every prefix ending at a dot
            for m in re.finditer(r"\.", k):
                p = k[: m.start() + 1]  # keep the trailing dot
                if k[len(p) :] in ref_keys:
                    prefix_counter[p] += 1
                    # shortest prefix that works is good enough
                    break

        if not prefix_counter:
            return src_state  # nothing matched → keep as-is

        # Use the prefix that matched the *most* keys
        prefix, _ = prefix_counter.most_common(1)[0]

    # Actually build a new state-dict with the prefix removed
    stripped_state = {
        (k[len(prefix) :] if k.startswith(prefix) else k): v
        for k, v in src_state.items()
    }
    return stripped_state
