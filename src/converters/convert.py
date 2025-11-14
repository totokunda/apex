import torch
from safetensors.torch import load_file
import pathlib
import os
from pydash import get
import os
import safetensors
import glob
from src.vae import get_vae
from typing import List, Dict, Any
from mlx.utils import tree_flatten

from src.converters.transformer_converters import (
    WanTransformerConverter,
    WanVaceTransformerConverter,
    WanMultiTalkTransformerConverter,
    CogVideoXTransformerConverter,
    HunyuanTransformerConverter,
    MochiTransformerConverter,
    LTXTransformerConverter,
    StepVideoTransformerConverter,
    SkyReelsTransformerConverter,
    HunyuanAvatarTransformerConverter,
    MagiTransformerConverter,
    FluxTransformerConverter,
    NoOpTransformerConverter,
)

from src.converters.utils import (
    get_model_class,
    get_empty_model,
    strip_common_prefix,
)

from src.converters.vae_converters import LTXVAEConverter, MagiVAEConverter


def get_transformer_converter(model_base: str):
    if (
        model_base == "wan.base"
        or model_base == "wan.causal"
        or model_base == "wan.fun"
        or model_base == "wan.recam"
    ):
        return WanTransformerConverter()
    elif model_base == "wan.vace":
        return WanVaceTransformerConverter()
    elif model_base == "wan.multitalk":
        return WanMultiTalkTransformerConverter()
    elif model_base == "cogvideox.base":
        return CogVideoXTransformerConverter()
    elif model_base == "hunyuan.base":
        return HunyuanTransformerConverter()
    elif model_base == "hunyuan.avatar":
        return HunyuanAvatarTransformerConverter()
    elif model_base == "mochi.base":
        return MochiTransformerConverter()
    elif model_base == "ltx.base":
        return LTXTransformerConverter()
    elif model_base == "stepvideo.base":
        return StepVideoTransformerConverter()
    elif model_base == "skyreels.base":
        return SkyReelsTransformerConverter()
    elif model_base == "magi.base":
        return MagiTransformerConverter()
    elif model_base == "flux.base":
        return FluxTransformerConverter()
    else:
        return NoOpTransformerConverter()


def get_transformer_converter_by_model_name(model_name: str):
    if "WanVace" in model_name:
        return WanVaceTransformerConverter()
    elif "WanMultiTalk" in model_name:
        return WanMultiTalkTransformerConverter()
    elif "Wan" in model_name:
        return WanTransformerConverter()
    elif "CogVideoX" in model_name:
        return CogVideoXTransformerConverter()
    elif "HunyuanAvatar" in model_name:
        return HunyuanAvatarTransformerConverter()
    elif "Hunyuan" in model_name:
        return HunyuanTransformerConverter()
    elif "Mochi" in model_name:
        return MochiTransformerConverter()
    elif "LTX" in model_name:
        return LTXTransformerConverter()
    elif "StepVideo" in model_name:
        return StepVideoTransformerConverter()
    elif "SkyReels" in model_name:
        return SkyReelsTransformerConverter()
    elif "Magi" in model_name:
        return MagiTransformerConverter()
    elif "Flux" in model_name or "Chroma" in model_name:
        return FluxTransformerConverter()
    return None


def get_vae_converter(vae_type: str, **additional_kwargs):
    if vae_type == "ltx":
        return LTXVAEConverter(**additional_kwargs)
    elif vae_type == "magi":
        return MagiVAEConverter()
    else:
        raise ValueError(f"VAE type {vae_type} not supported")


def load_safetensors(dir: pathlib.Path):
    """Load a sharded safetensors file."""
    # load all shards
    # check if single shard
    if dir.is_file() and dir.suffix == ".safetensors":
        return load_file(dir)

    shards = list(dir.glob("*.safetensors"))
    if len(shards) == 0:
        raise ValueError(f"No shards found in {dir}")

    state_dict = {}
    for shard in shards:
        shard_state_dict = load_file(shard)
        state_dict.update(shard_state_dict)
    return state_dict


def load_pt(dir: pathlib.Path):
    """Load a sharded pt file."""
    pt_extensions = tuple(["pt", "bin", "pth"])
    if dir.is_file() and dir.suffix.endswith(pt_extensions):
        return torch.load(dir, weights_only=True, map_location="cpu", mmap=True)

    shards = list(dir.glob(f"*{pt_extensions}"))
    if len(shards) == 0:
        raise ValueError(f"No shards found in {dir}")

    state_dict = {}
    for shard in shards:
        shard_state_dict = torch.load(
            shard, weights_only=True, map_location="cpu", mmap=True
        )
        state_dict.update(shard_state_dict)
    return state_dict


def is_safetensors_file(file_path: str):
    try:
        with safetensors.safe_open(file_path, framework="pt", device="cpu") as f:
            f.keys()
        return True
    except Exception:
        return False


def load_state_dict(ckpt_path: str, model_key: str = None, pattern: str | None = None):
    pt_extensions = tuple(["pt", "bin", "pth", "ckpt"])
    state_dict = {}
    if is_safetensors_file(ckpt_path):
        state_dict = load_safetensors(pathlib.Path(ckpt_path))
    elif ckpt_path.endswith(pt_extensions):
        state_dict = load_pt(pathlib.Path(ckpt_path))
    elif os.path.isdir(ckpt_path):
        files = []
        format = None
        if pattern is None:
            files = glob.glob(os.path.join(ckpt_path, "*.safetensors"))
            if len(files) == 0:
                files = (
                    glob.glob(os.path.join(ckpt_path, "*.pt"))
                    + glob.glob(os.path.join(ckpt_path, "*.bin"))
                    + glob.glob(os.path.join(ckpt_path, "*.pth"))
                    + glob.glob(os.path.join(ckpt_path, "*.ckpt"))
                )
                format = "pt"
            else:
                format = "safetensors"
        else:
            files = glob.glob(os.path.join(ckpt_path, pattern))
            # if all files are safetensors, set format to safetensors
            if all(is_safetensors_file(file) for file in files):
                format = "safetensors"
            # if all files are pt, set format to pt
            elif all(file.endswith(pt_extensions) for file in files):
                format = "pt"

        if len(files) == 0:
            raise ValueError(f"No files found in {ckpt_path} with pattern {pattern}")

        if format == "safetensors":
            state_dict = load_safetensors(pathlib.Path(ckpt_path))
        elif format == "pt":
            state_dict = load_pt(pathlib.Path(ckpt_path))
        else:
            for file in files:
                if is_safetensors_file(file):
                    state_dict.update(load_safetensors(pathlib.Path(file)))
                elif file.endswith(pt_extensions):
                    state_dict.update(load_pt(pathlib.Path(file)))
                else:
                    raise ValueError(f"Unsupported checkpoint format: {file}")
    else:
        raise ValueError(f"Unsupported checkpoint format: {ckpt_path}")

    if len(state_dict.keys()) == 1:
        return state_dict[next(iter(state_dict.keys()))]

    if model_key is not None:
        state_dict = get(state_dict, model_key)

    return state_dict


def needs_conversion(
    original_state_dict: Dict[str, Any],
    state_dict: Dict[str, Any],
    keys_to_ignore: List[str] = [],
):
    if not keys_to_ignore:
        keys_to_ignore = []
    for key in state_dict.keys():
        if any(ignore_key in key for ignore_key in keys_to_ignore):
            continue
        if (
            key not in original_state_dict
            or original_state_dict[key].shape != state_dict[key].shape
        ):
            return True
    return False


def remove_keys_from_state_dict(
    state_dict: Dict[str, Any], keys_to_ignore: List[str] = []
):
    keys = list(state_dict.keys())
    if not keys_to_ignore:
        keys_to_ignore = []
    for key in keys:
        if any(ignore_key in key for ignore_key in keys_to_ignore):
            state_dict.pop(key)
    return state_dict


def convert_transformer(
    config: dict,
    model_base: str,
    ckpt_path: str | List[str] = None,
    model_key: str = None,
    pattern: str | None = None,
    **transformer_converter_kwargs,
):

    if "mlx" in model_base:
        using_mlx = True
        model_base = model_base.replace("mlx.", "")
        model_type = "mlx.transformer"
    else:
        using_mlx = False
        model_type = "transformer"

    model_class = get_model_class(model_base, config, model_type=model_type)

    converter = get_transformer_converter(model_base)

    model = get_empty_model(model_class, config)

    original_state_dict = model.state_dict()

    keys_to_ignore = getattr(model, "_keys_to_ignore_on_load_unexpected", [])

    if isinstance(ckpt_path, list):
        state_dict = {}
        for i, ckpt in enumerate(ckpt_path):
            ct_state_dict = load_state_dict(ckpt, model_key, pattern)
            if needs_conversion(original_state_dict, ct_state_dict, keys_to_ignore):
                converter.convert(ct_state_dict)
            state_dict.update(ct_state_dict)
    else:
        state_dict = load_state_dict(ckpt_path, model_key, pattern)
        if needs_conversion(original_state_dict, state_dict, keys_to_ignore):
            converter.convert(state_dict)

    state_dict = remove_keys_from_state_dict(state_dict, keys_to_ignore)

    if not using_mlx:
        state_dict = strip_common_prefix(state_dict, model.state_dict())
        model.load_state_dict(state_dict, strict=True, assign=True)
    else:
        model_state_dict = tree_flatten(model.parameters(), destination={})
        state_dict = strip_common_prefix(state_dict, model_state_dict)
        model.load_weights(state_dict)

    return model


def convert_vae(
    config: dict,
    vae_type: str,
    ckpt_path: str | List[str] = None,
    model_key: str = None,
    pattern: str | None = None,
    **vae_converter_kwargs,
):
    model_class = get_model_class(vae_type, config, model_type="vae")
    model = get_empty_model(model_class, config)
    original_state_dict = model.state_dict()
    keys_to_ignore = getattr(model, "_keys_to_ignore_on_load_unexpected", [])

    converter = get_vae_converter(vae_type, **vae_converter_kwargs)
    if isinstance(ckpt_path, list):
        state_dict = {}
        for ckpt in ckpt_path:
            ct_state_dict = load_state_dict(ckpt, model_key, pattern)
            if needs_conversion(original_state_dict, ct_state_dict, keys_to_ignore):
                converter.convert(ct_state_dict)
            state_dict.update(ct_state_dict)
    else:
        state_dict = load_state_dict(ckpt_path, model_key, pattern)
        if needs_conversion(original_state_dict, state_dict, keys_to_ignore):
            converter.convert(state_dict)

    state_dict = remove_keys_from_state_dict(state_dict, keys_to_ignore)
    state_dict = strip_common_prefix(state_dict, model.state_dict())

    model.load_state_dict(state_dict, strict=True, assign=True)

    return model


def get_transformer_keys(model_base: str, config: dict):

    using_mlx = False
    if "mlx" in model_base:
        using_mlx = True
        model_base = model_base.replace("mlx.", "")
        model_type = "mlx.transformer"
    else:
        model_type = "transformer"

    model_class = get_model_class(model_base, config, model_type=model_type)
    model = get_empty_model(model_class, config)

    if using_mlx:
        params = model.parameters()
        flat_params = tree_flatten(params, destination={})
        return flat_params.keys()
    else:
        return model.state_dict().keys()


def get_vae_keys(vae_type: str, config: dict):
    if vae_type != "ltx":
        return []
    model_class = get_vae(vae_type)

    model = get_empty_model(model_class, config)
    return model.state_dict().keys()


if __name__ == "__main__":
    model = convert_transformer(
        "wan_t2v_14b", "wan", "/mnt/localssd/Wan14BT2VFusioniX_fp16_.safetensors"
    )
