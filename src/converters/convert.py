import torch
from safetensors.torch import load_file
import pathlib
import os
from pydash import get
import os
import safetensors
import glob
from src.transformer.base import get_transformer
from src.vae import get_vae
from accelerate import init_empty_weights
from typing import List

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
)

from src.converters.utils import (
    get_transformer_config,
    get_vae_config,
    get_model_class,
    get_empty_model,
    strip_common_prefix,
)

from src.converters.vae_converters import LTXVAEConverter, MagiVAEConverter


def get_transformer_converter(model_type: str):
    if (
        model_type == "wan.base"
        or model_type == "wan.causal"
        or model_type == "wan.fun"
    ):
        return WanTransformerConverter()
    elif model_type == "wan.vace":
        return WanVaceTransformerConverter()
    elif model_type == "wan.multitalk":
        return WanMultiTalkTransformerConverter()
    elif model_type == "cogvideox.base":
        return CogVideoXTransformerConverter()
    elif model_type == "hunyuan.base":
        return HunyuanTransformerConverter()
    elif model_type == "hunyuan.avatar":
        return HunyuanAvatarTransformerConverter()
    elif model_type == "mochi.base":
        return MochiTransformerConverter()
    elif model_type == "ltx.base":
        return LTXTransformerConverter()
    elif model_type == "stepvideo.base":
        return StepVideoTransformerConverter()
    elif model_type == "skyreels.base":
        return SkyReelsTransformerConverter()
    elif model_type == "magi.base":
        return MagiTransformerConverter()
    else:
        raise ValueError(f"Model type {model_type} not supported")


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
    pt_extensions = tuple(["pt", "bin", "pth"])
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


def convert_transformer(
    model_tag: str,
    model_type: str,
    ckpt_path: str | List[str] = None,
    model_key: str = None,
    pattern: str | None = None,
    **transformer_converter_kwargs,
):

    config = get_transformer_config(
        model_tag, config_path=transformer_converter_kwargs.get("config_path", None)
    )

    model_class = get_model_class(model_type, config, model_type="transformer")

    converter = get_transformer_converter(model_type)

    if isinstance(ckpt_path, list):
        state_dict = {}
        for ckpt in ckpt_path:
            state_dict.update(load_state_dict(ckpt, model_key, pattern))
    else:
        state_dict = load_state_dict(ckpt_path, model_key, pattern)

    model = get_empty_model(model_class, config)

    model_keys = model.state_dict().keys()
    # print(model_keys)
    # exit()

    converter.convert(state_dict)

    state_dict = strip_common_prefix(state_dict, model.state_dict())

    model.load_state_dict(state_dict, strict=True, assign=True)

    return model


def convert_vae(
    vae_tag: str,
    vae_type: str,
    ckpt_path: str | List[str] = None,
    model_key: str = None,
    pattern: str | None = None,
    **vae_converter_kwargs,
):
    config = get_vae_config(
        vae_tag, config_path=vae_converter_kwargs.get("config_path", None)
    )
    model_class = get_model_class(vae_type, config, model_type="vae")

    converter = get_vae_converter(vae_type, **vae_converter_kwargs)
    if isinstance(ckpt_path, list):
        state_dict = {}
        for ckpt in ckpt_path:
            state_dict.update(load_state_dict(ckpt, model_key, pattern))
    else:
        state_dict = load_state_dict(ckpt_path, model_key, pattern)

    model = get_empty_model(model_class, config)

    converter.convert(state_dict)

    state_dict = strip_common_prefix(state_dict, model.state_dict())

    model.load_state_dict(state_dict, strict=True, assign=True)

    return model


def get_transformer_keys(
    model_type: str, model_tag: str, transformer_converter_kwargs: dict
):
    config = get_transformer_config(
        model_tag, config_path=transformer_converter_kwargs.get("config_path", None)
    )
    model_class = get_model_class(model_type, config, model_type="transformer")
    model = get_empty_model(model_class, config)
    return model.state_dict().keys()


def get_vae_keys(vae_type: str, vae_tag: str, vae_converter_kwargs: dict):
    if vae_type != "ltx":
        return []
    model_class = get_vae(vae_type)
    config = get_vae_config(
        vae_tag, config_path=vae_converter_kwargs.get("config_path", None)
    )
    model = get_empty_model(model_class, config)
    return model.state_dict().keys()


if __name__ == "__main__":
    model = convert_transformer(
        "wan_t2v_14b", "wan", "/mnt/localssd/Wan14BT2VFusioniX_fp16_.safetensors"
    )
