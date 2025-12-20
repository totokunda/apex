from safetensors.torch import safe_open
import torch
import torch.nn as nn
from src.transformer.wan.base.model import WanTransformer3DModel
from accelerate import init_empty_weights
from transformers import UMT5EncoderModel, PretrainedConfig
import json
import os
from glob import glob
import time
from tqdm import tqdm
# Maximize CPU parallelism

path = "/home/tosin_coverquick_co/apex-diffusion/components/Wan-AI_Wan2.2-T2V-A14B-Diffusers/transformer"
target_dtype = torch.float16
load_device = "cuda" if torch.cuda.is_available() else "cpu"

start_time = time.perf_counter()

# Pre-index modules for fast name -> module resolution.
def _build_module_index(root: nn.Module) -> dict[str, nn.Module]:
    # named_modules includes the root with name == ""
    return dict(root.named_modules())


def _set_tensor_by_fqn(modules: dict[str, nn.Module], fqn: str, tensor: torch.Tensor) -> bool:
    """
    Fast, shard-streaming weight load:
    - Avoids building a full shard dict in RAM
    - Avoids load_state_dict traversal overhead
    - Replaces meta parameters/buffers with real tensors as we go
    """
    if "." in fqn:
        prefix, attr = fqn.rsplit(".", 1)
    else:
        prefix, attr = "", fqn

    module = modules.get(prefix, None)
    if module is None:
        return False

    # Parameters
    if attr in module._parameters:
        # Replace meta Parameter with a real one; keep no-grad semantics.
        module._parameters[attr] = nn.Parameter(tensor, requires_grad=False)
        return True

    # Buffers
    if attr in module._buffers:
        module._buffers[attr] = tensor
        return True

    # Fallback: setattr for odd cases (rare); still avoids load_state_dict.
    try:
        setattr(module, attr, tensor)
        return True
    except Exception:
        return False


# Create empty model
config = json.load(open(os.path.join(path, "config.json")))
with init_empty_weights():
    model = WanTransformer3DModel.from_config(config)

filenames = glob(os.path.join(path, "*.safetensors"))
filenames.sort()
modules = _build_module_index(model)

missing = 0
total = 0
stream = torch.cuda.Stream()
with torch.no_grad():
    for filename in tqdm(filenames, desc=f"Loading model (streaming, device={load_device})"):
        # Stream tensors directly from shard -> model, avoiding an intermediate dict.
        with safe_open(filename=filename, framework="pt", device="cpu") as f:
            for name in f.keys():
                total += 1
                t = f.get_tensor(name)
                with torch.cuda.stream(stream):
                    if t.dtype != target_dtype:
                        t = t.to(dtype=target_dtype, copy=False)
                    t = t.to(device=load_device, non_blocking=True)
                ok = _set_tensor_by_fqn(modules, name, t)
                if not ok:
                    missing += 1

        # Reduce peak allocator fragmentation when loading huge models on GPU.
        if load_device == "cuda":
            torch.cuda.empty_cache()
    torch.cuda.synchronize()

print(time.perf_counter() - start_time, "seconds")
if missing:
    print(f"Warning: {missing}/{total} tensors did not match a param/buffer name on the model.")
