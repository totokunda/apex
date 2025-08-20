import torch
import torch.nn as nn
import torch.nn.functional as F
import gguf
from typing import Optional, Tuple

from src.quantize.dequant import is_quantized, dequantize_tensor
from src.quantize.ggml_tensor import GGMLTensor


def cast_to(
    t: Optional[torch.Tensor],
    dtype: Optional[torch.dtype],
    device,
    copy=False,
    non_blocking=False,
):
    if t is None:
        return None
    if (dtype is None or t.dtype == dtype) and (t.device == device):
        return t
    return t.to(
        device=device,
        dtype=dtype if dtype is not None else t.dtype,
        copy=copy,
        non_blocking=non_blocking,
    )


class GGMLLayer(nn.Module):
    """
    Base mixin for GGML-backed layers that need on-the-fly dequantization.
    """

    dequant_dtype: Optional[torch.dtype] = (
        None  # preferred dequant dtype (fallback to input/bfloat16/fp16)
    )
    largest_layer: bool = False

    torch_compatible_tensor_types = {
        None,
        gguf.GGMLQuantizationType.F32,
        gguf.GGMLQuantizationType.F16,
    }

    # -------------------- quantization helpers --------------------

    def is_ggml_quantized(self, *, weight=None, bias=None) -> bool:
        if weight is None:
            weight = getattr(self, "weight", None)
        if bias is None:
            bias = getattr(self, "bias", None)
        return is_quantized(weight) or is_quantized(bias)

    def _effective_dequant_dtype(
        self, requested: Optional[torch.dtype], tensor: Optional[torch.Tensor]
    ) -> torch.dtype:
        # priority: explicit requested dtype -> layer.default -> tensor.dequant_dtype -> torch.float16

        if requested is not None:
            return requested
        if getattr(self, "dequant_dtype", None) is not None:
            return self.dequant_dtype
        if (
            isinstance(tensor, GGMLTensor)
            and getattr(tensor, "dequant_dtype", None) is not None
        ):
            return tensor.dequant_dtype
        # sensible default for speed/memory (change to float32 if you prefer accuracy)
        return torch.float16

    def _materialize_weight(
        self, t: Optional[torch.Tensor], *, target_dtype: Optional[torch.dtype], device
    ) -> Optional[torch.Tensor]:
        if t is None:
            return None

        # Non-quantized GGML (F16/F32) or plain torch tensor → just cast
        if not is_quantized(t):
            out = t
        else:
            # Quantized → dequantize first
            dq_dtype = self._effective_dequant_dtype(target_dtype, t)
            # dequantize_tensor interfaces differ across repos; prefer (tensor, out_dtype),
            # and fall back to (tensor, out_dtype, dq_dtype_hint) if available.
            try:
                out = dequantize_tensor(
                    t, dq_dtype
                )  # common signature: (tensor, out_dtype)
            except TypeError:
                out = dequantize_tensor(t, dq_dtype, getattr(t, "dequant_dtype", None))

        # Final device/dtype cast (no-op if already right)
        out = cast_to(out, target_dtype, device, copy=False)
        return out

    def cast_bias_weight(
        self,
        input: Optional[torch.Tensor] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        bias_dtype: Optional[torch.dtype] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Infer dtype/device from input if not provided
        if input is not None:
            device = input.device if device is None else device
            if dtype is None:
                dtype = input.dtype

        # Weight
        weight = self._materialize_weight(
            getattr(self, "weight", None), target_dtype=dtype, device=device
        )

        # Bias (if present)
        bias = None
        if hasattr(self, "bias") and self.bias is not None:
            bdtype = (
                bias_dtype
                if bias_dtype is not None
                else (
                    dtype
                    if dtype is not None
                    else self._effective_dequant_dtype(None, self.bias)
                )
            )
            bias = self._materialize_weight(
                self.bias, target_dtype=bdtype, device=device
            )

        return weight, bias

    # -------------------- state_dict I/O --------------------

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # if any GGML quant present or this is Linear/large Embedding, route to custom loader
        weight = state_dict.get(f"{prefix}weight", None)
        bias = state_dict.get(f"{prefix}bias", None)

        should_route = self.is_ggml_quantized(weight=weight, bias=bias) or isinstance(
            self, nn.Linear
        )
        if (
            not should_route
            and isinstance(self, nn.Embedding)
            and getattr(self, "weight", torch.empty(0)).shape[0] >= (64 * 1024)
        ):
            should_route = True

        if should_route:
            return self.ggml_load_from_state_dict(state_dict, prefix, *args, **kwargs)

        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def ggml_load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        # only consider keys that actually start with this prefix
        pfx = str(prefix)
        pfx_len = len(pfx)

        got_weight = False
        got_bias = False

        for k, v in state_dict.items():
            if not k.startswith(pfx):
                continue
            suffix = k[pfx_len:]
            if suffix == "weight":
                self.weight = nn.Parameter(v, requires_grad=False)
                got_weight = True
            elif suffix == "bias":
                if v is not None:
                    self.bias = nn.Parameter(v, requires_grad=False)
                got_bias = True
            else:
                unexpected_keys.append(k)

        # If linear is missing weight, create a placeholder of correct shape
        if not got_weight and isinstance(self, nn.Linear):
            # Correct ordering: (out_features, in_features)
            w = torch.zeros(self.out_features, self.in_features, dtype=torch.float32)
            self.weight = nn.Parameter(w, requires_grad=False)
            missing_keys.append(prefix + "weight")

        # Mark for VRAM estimation if needed
        if getattr(self.weight, "is_largest_weight", False):
            self.largest_layer = True

    def _save_to_state_dict(self, *args, **kwargs):
        return self.ggml_save_to_state_dict(*args, **kwargs)

    def ggml_save_to_state_dict(self, destination, prefix, keep_vars):
        destination[prefix + "weight"] = self.weight
        if getattr(self, "bias", None) is not None:
            destination[prefix + "bias"] = self.bias


# -------------------- Patched layers --------------------


class GGMLLinear(GGMLLayer, nn.Linear):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w, b = self.cast_bias_weight(x)
        return F.linear(x, w, b)


class GGMLConv2d(GGMLLayer, nn.Conv2d):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w, b = self.cast_bias_weight(x)
        return F.conv2d(x, w, b, self.stride, self.padding, self.dilation, self.groups)


class GGMLConv1d(GGMLLayer, nn.Conv1d):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w, b = self.cast_bias_weight(x)
        return F.conv1d(x, w, b, self.stride, self.padding, self.dilation, self.groups)


class GGMLEmbedding(GGMLLayer, nn.Embedding):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use input's device/dtype; bias is irrelevant for Embedding
        w, _ = self.cast_bias_weight(
            x,
            dtype=self.weight.dtype if hasattr(self, "weight") else None,
            device=x.device,
        )
        return F.embedding(
            x,
            w,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )


class GGMLLayerNorm(GGMLLayer, nn.LayerNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w, b = self.cast_bias_weight(x)
        return F.layer_norm(x, self.normalized_shape, w, b, self.eps)


class GGMLGroupNorm(GGMLLayer, nn.GroupNorm):
    def __init__(self, *args, dequant_dtype: Optional[torch.dtype] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.dequant_dtype = dequant_dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w, b = self.cast_bias_weight(x)
        return F.group_norm(x, self.num_groups, w, b, self.eps)


_TYPE_MAP = {
    nn.Linear: GGMLLinear,
    nn.Conv2d: GGMLConv2d,
    nn.Conv1d: GGMLConv1d,
    nn.Embedding: GGMLEmbedding,
    nn.LayerNorm: GGMLLayerNorm,
    nn.GroupNorm: GGMLGroupNorm,
}


def patch_model(
    model: nn.Module,
    name_filter=None,
    *,
    default_dequant_dtype: Optional[torch.dtype] = None,
):
    """
    In-place class swap. If name_filter is provided, only patch qualified names
    for which name_filter(qname) == True.
    Optionally set a global default dequant dtype for all patched modules.
    """
    stack = [("", model)]
    while stack:
        prefix, mod = stack.pop()
        for name, child in mod._modules.items():
            qname = f"{prefix}{name}"
            t = type(child)
            if t in _TYPE_MAP and (name_filter is None or name_filter(qname)):
                child.__class__ = _TYPE_MAP[t]  # swap class w/o reallocation
                if default_dequant_dtype is not None and isinstance(child, GGMLLayer):
                    child.dequant_dtype = default_dequant_dtype
            if child is not None and len(child._modules) > 0:
                stack.append((qname + ".", child))
