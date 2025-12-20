"""
Ideas to squeeze more load speed out of the streaming loader in mo.py without raising VRAM/RAM.
- Overlap disk reads with H2D copies via a bounded prefetch queue and a dedicated CUDA stream.
- Avoid repeated string parsing and module lookups by caching FQN -> (module, attr, is_param, is_buffer).
Both approaches target CPU overhead and PCIe idle time; peak memory stays bounded because the queue is size 1.
"""

from __future__ import annotations

import os
import queue
import threading
from glob import glob
from typing import Dict, Iterable, Iterator, Tuple
import json
import torch
import torch.nn as nn
from accelerate import init_empty_weights
from safetensors.torch import safe_open
from transformers import PretrainedConfig, UMT5EncoderModel
from src.transformer.wan.base.model import WanTransformer3DModel

def _build_setter_table(root: nn.Module) -> Dict[str, Tuple[nn.Module, str, bool, bool]]:
    """
    Precompute where each FQN lands to sidestep string splitting and dict lookups in the hot path.
    Returns a map: fqn -> (module, attr, is_param, is_buffer).
    """
    table: Dict[str, Tuple[nn.Module, str, bool, bool]] = {}
    for fqn, module in root.named_modules():
        for name, _ in module._parameters.items():
            table[f"{fqn}.{name}" if fqn else name] = (module, name, True, False)
        for name, _ in module._buffers.items():
            table[f"{fqn}.{name}" if fqn else name] = (module, name, False, True)
    return table


def _set_from_table(
    setters: Dict[str, Tuple[nn.Module, str, bool, bool]],
    fqn: str,
    tensor: torch.Tensor,
) -> bool:
    entry = setters.get(fqn)
    if entry is None:
        return False
    module, attr, is_param, is_buffer = entry
    if is_param:
        module._parameters[attr] = nn.Parameter(tensor, requires_grad=False)
        return True
    if is_buffer:
        module._buffers[attr] = tensor
        return True
    setattr(module, attr, tensor)
    return True


def iter_prefetched_tensors(
    filenames: Iterable[str],
    target_dtype: torch.dtype,
    device: str,
) -> Iterator[Tuple[str, torch.Tensor]]:
    """
    Stream tensors with a single-element queue to overlap disk IO and H2D copies.
    Bounded queue keeps RAM usage flat while allowing the reader thread to stay ahead.
    """
    q: "queue.Queue[tuple[str, torch.Tensor] | object]" = queue.Queue(maxsize=1)
    sentinel = object()

    def reader() -> None:
        for filename in filenames:
            # CPU target keeps GPU free while the main thread consumes; pin for faster copies.
            with safe_open(filename=filename, framework="pt", device=device) as f:
                for name in f.keys():
                    t = f.get_tensor(name)
                    if t.dtype != target_dtype:
                        t = t.to(dtype=target_dtype, copy=False)
                    q.put((name, t))
        q.put(sentinel)

    threading.Thread(target=reader, daemon=True).start()

    copy_stream = torch.cuda.Stream() if device.startswith("cuda") else None
    while True:
        item = q.get()
        if item is sentinel:
            break
        name, tensor = item
        if copy_stream and tensor.device.type == "cpu":
            with torch.cuda.stream(copy_stream):
                tensor = tensor.to(device, non_blocking=True)
        elif tensor.device.type != device:
            tensor = tensor.to(device, non_blocking=True)
        yield name, tensor

    if copy_stream:
        copy_stream.synchronize()


def load_with_prefetch(path: str, device: str = "cuda", target_dtype: torch.dtype = torch.bfloat16) -> WanTransformer3DModel:
    """
    Prototype loader:
    - Uses prefetch + dedicated CUDA stream to overlap IO and copies.
    - Uses cached setters to avoid per-tensor string parsing.
    """
    config = json.load(open(os.path.join(path, "config.json")))
    with init_empty_weights():
        model = WanTransformer3DModel.from_config(config)

    setters = _build_setter_table(model)
    filenames = sorted(glob(os.path.join(path, "*.safetensors")))

    for name, tensor in iter_prefetched_tensors(filenames, target_dtype, device):
        _set_from_table(setters, name, tensor)

    if device.startswith("cuda"):
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    return model


if __name__ == "__main__":
    import time
    start_time = time.perf_counter()
    # Example usage (path should point to the text_encoder shards).
    model_path = "/home/tosin_coverquick_co/apex-diffusion/components/Wan-AI_Wan2.2-T2V-A14B-Diffusers/transformer"
    model = load_with_prefetch(model_path, device="cuda")
    print("Model loaded (prefetch prototype).")
    print(time.perf_counter() - start_time, "seconds")