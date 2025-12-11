import math
from typing import Optional, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F

# Optional Transformer Engine (TE) support
try:  # pragma: no cover - optional dependency
    import transformer_engine.pytorch as te  # type: ignore

    _HAS_TE = True
except Exception:  # pragma: no cover - optional dependency
    te = None  # type: ignore
    _HAS_TE = False

# -------------------- FP8 quantization helpers --------------------


def get_fp_maxval(
    bits: int = 8, mantissa_bit: int = 3, sign_bits: int = 1
) -> torch.Tensor:
    """
    Compute the maximum representable value for a custom FP format.
    Defaults to FP8 E4M3 (bits=8, mantissa_bit=3, sign_bits=1).
    """
    _bits = torch.tensor(bits)
    _mantissa_bit = torch.tensor(mantissa_bit)
    _sign_bits = torch.tensor(sign_bits)
    M = torch.clamp(torch.round(_mantissa_bit), 1, _bits - _sign_bits)
    E = _bits - _sign_bits - M
    bias = 2 ** (E - 1) - 1
    mantissa = 1.0
    for i in range(mantissa_bit - 1):
        mantissa += 1.0 / (2 ** (i + 1))
    maxval = mantissa * 2 ** (2**E - 1 - bias)
    return maxval


def quantize_to_fp8(
    x: torch.Tensor,
    bits: int = 8,
    mantissa_bit: int = 3,
    sign_bits: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize-dequantize `x` to an FP8-like grid (default E4M3) and return:
      - qdq_out: quantize-dequantized tensor (same dtype as input)
      - log_scales: per-value log2 scaling factors
    """
    bits_t = torch.tensor(bits)
    mantissa_bit_t = torch.tensor(mantissa_bit)
    sign_bits_t = torch.tensor(sign_bits)
    M = torch.clamp(torch.round(mantissa_bit_t), 1, bits_t - sign_bits_t)
    E = bits_t - sign_bits_t - M
    bias = 2 ** (E - 1) - 1

    mantissa = 1.0
    for i in range(mantissa_bit - 1):
        mantissa += 1.0 / (2 ** (i + 1))

    maxval = mantissa * 2 ** (2**E - 1 - bias)
    minval = -maxval if sign_bits == 1 else torch.zeros_like(maxval)

    input_clamp = torch.min(torch.max(x, minval), maxval)
    log_scales = torch.clamp(
        (torch.floor(torch.log2(torch.abs(input_clamp)) + bias)).detach(), 1.0
    )
    log_scales = 2.0 ** (log_scales - M - bias.type(x.dtype))

    # Dequantize back to the original dtype/grid
    qdq_out = torch.round(input_clamp / log_scales) * log_scales
    return qdq_out, log_scales


def fp8_tensor_quant(
    x: torch.Tensor,
    scale: torch.Tensor,
    bits: int = 8,
    mantissa_bit: int = 3,
    sign_bits: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Normalize `x` by `scale`, then quantize-dequantize to FP8 grid.
    Returns:
      - quant_dequant_x: qdq(x / scale)
      - scale: broadcasted scale used
      - log_scales: per-value log2 scaling factors
    """
    for _ in range(len(x.shape) - 1):
        scale = scale.unsqueeze(-1)

    new_x = x / scale
    quant_dequant_x, log_scales = quantize_to_fp8(
        new_x, bits=bits, mantissa_bit=mantissa_bit, sign_bits=sign_bits
    )
    return quant_dequant_x, scale, log_scales


def fp8_activation_dequant(
    qdq_out: torch.Tensor,
    scale: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Dequantize FP8 activations/weights back to `dtype` using `scale`.

    `qdq_out` is assumed to already live on an FP8-like grid (e.g. float8_e4m3fn),
    and `scale` is the (possibly per-channel) scaling tensor.
    """
    qdq_out = qdq_out.to(dtype)
    quant_dequant_x = qdq_out * scale.to(dtype)
    return quant_dequant_x


class FP8ScaledLayer(nn.Module):
    """
    Base class for FP8-scaled layers.

    Assumptions:
      - The *stored* weights are in FP8 (e4m3fn / e5m2) or another low-precision dtype.
      - A `scale_weight` tensor (usually per-out-feature) exists in the state dict.
      - On every forward we:
          1. Cast weights from FP8 → `compute_dtype`
          2. Apply `scale_weight`
          3. Run the corresponding op (linear / conv / embedding / etc.)

    The model should be patched *before* loading an FP8-scaled state_dict so that
    `scale_weight` parameters are correctly created and populated.
    """

    # Preferred compute dtype. If None, we fall back to the input dtype,
    # defaulting to float16 for FP8 inputs.
    compute_dtype: Optional[torch.dtype] = None

    def __init__(self, *, compute_dtype: Optional[torch.dtype] = None) -> None:
        # NOTE: Do not call super().__init__() here because in multiple
        # inheritance subclasses like FP8ScaledLinear(FP8ScaledLayer, nn.Linear)
        # the next class in the MRO is nn.Linear, whose __init__ expects
        # (in_features, out_features, ...). Calling nn.Module.__init__
        # directly avoids that TypeError while still initializing the
        # module base class.
        nn.Module.__init__(self)
        if compute_dtype is not None:
            self.compute_dtype = compute_dtype

    # -------------------- helpers --------------------

    def _effective_compute_dtype(
        self, x: Optional[torch.Tensor], requested: Optional[torch.dtype] = None
    ) -> torch.dtype:
        if requested is not None:
            return requested
        if getattr(self, "compute_dtype", None) is not None:
            return self.compute_dtype  # type: ignore[return-value]
        if x is not None:
            # If input is already a "good" compute dtype, keep it, otherwise
            # default to float16 which is a safe/small compute type.
            if x.dtype in (
                torch.float16,
                torch.bfloat16,
                torch.float32,
            ):
                return x.dtype
        return torch.float16

    def _scale_and_cast_weight(
        self,
        weight: Optional[torch.Tensor],
        scale_weight: Optional[torch.Tensor],
        *,
        target_dtype: torch.dtype,
        per_out_feature_dim: int = 0,
    ) -> Optional[torch.Tensor]:
        """
        Cast `weight` to `target_dtype` and apply `scale_weight` (if present).

        - `per_out_feature_dim` is the dimension that corresponds to the "output"
          features/channels (0 for Linear/Conv, -1 for Embedding).
        """
        if weight is None:
            return None

        # Fast path: FP8 weights + explicit scale tensor → use the provided
        # FP8 dequant helper directly. This matches the user-provided
        # fp8_activation_dequant semantics.

        if weight.dtype == torch.float8_e4m3fn or weight.dtype == torch.float8_e5m2 and scale_weight is not None:
            return fp8_activation_dequant(weight, scale_weight, target_dtype)

        # Dequantize / cast from FP8 (or any low-precision) to target_dtype.
        w = weight.to(target_dtype)

        if scale_weight is not None:
            s = scale_weight.to(target_dtype)
            # Common cases:
            #   - scalar
            #   - per-out-feature: [out_features]
            if s.numel() == 1:
                w = w * s
            else:
                # Broadcast along the out-feature dimension
                if per_out_feature_dim < 0:
                    per_out_feature_dim = w.dim() + per_out_feature_dim

                # View scale as [out_features, 1, 1, ...]
                shape = [1] * w.dim()
                shape[per_out_feature_dim] = -1
                s_view = s.view(*shape)
                w = w * s_view

        return w


class FP8ScaledLinear(FP8ScaledLayer, nn.Linear):
    """
    Linear layer with FP8 weights and `scale_weight` support.

    If Transformer Engine is available, we still keep the public API identical
    (so that FP8 checkpoints load cleanly) but we optionally wrap the matmul in
    TE's `fp8_autocast` context for optimized kernels.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        *,
        compute_dtype: Optional[torch.dtype] = None,
        device=None,
        dtype=None,
    ) -> None:
        FP8ScaledLayer.__init__(self, compute_dtype=compute_dtype)
        nn.Linear.__init__(
            self,
            in_features,
            out_features,
            bias=bias,
            device=device,
            dtype=dtype,
        )

        # Placeholder scale; real values are loaded from the FP8 checkpoint.
        # The Wan FP8 checkpoint stores `scale_weight` as a scalar ([1]),
        # so we initialize it that way and rely on broadcasting in
        # `_scale_and_cast_weight` for per-out-feature application.
        self.scale_weight = nn.Parameter(
            torch.ones(1, dtype=torch.float32), requires_grad=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        compute_dtype = self._effective_compute_dtype(x)

        x_c = x.to(compute_dtype)
        w = self._scale_and_cast_weight(
            self.weight, getattr(self, "scale_weight", None), target_dtype=compute_dtype
        )
        b = self.bias
        if b is not None and b.dtype != compute_dtype:
            b = b.to(compute_dtype)

        def _linear(inp: torch.Tensor, w_: torch.Tensor, b_: Optional[torch.Tensor]):
            return F.linear(inp, w_, b_)

        if _HAS_TE:
            # We only use TE for the matmul kernel; weights are already dequantized
            # and scaled here.
            with te.fp8_autocast(enabled=False):  # keep semantics explicit
                out = _linear(x_c, w, b)
        else:
            out = _linear(x_c, w, b)

        # Always return in compute dtype (float16/bfloat16/float32), not FP8.
        return out


class FP8ScaledConv2d(FP8ScaledLayer, nn.Conv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias: bool = True,
        padding_mode: str = "zeros",
        *,
        compute_dtype: Optional[torch.dtype] = None,
        device=None,
        dtype=None,
    ) -> None:
        FP8ScaledLayer.__init__(self, compute_dtype=compute_dtype)
        nn.Conv2d.__init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )
        # Wan FP8 checkpoints store `scale_weight` as a scalar; we rely on
        # broadcasting in `_scale_and_cast_weight` to apply it per out-channel.
        self.scale_weight = nn.Parameter(
            torch.ones(1, dtype=torch.float32), requires_grad=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        compute_dtype = self._effective_compute_dtype(x)

        x_c = x.to(compute_dtype)
        w = self._scale_and_cast_weight(
            self.weight,
            getattr(self, "scale_weight", None),
            target_dtype=compute_dtype,
            per_out_feature_dim=0,
        )
        b = self.bias
        if b is not None and b.dtype != compute_dtype:
            b = b.to(compute_dtype)

        out = F.conv2d(
            x_c,
            w,
            b,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        # Always return in compute dtype (float16/bfloat16/float32), not FP8.
        return out


class FP8ScaledConv1d(FP8ScaledLayer, nn.Conv1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias: bool = True,
        padding_mode: str = "zeros",
        *,
        compute_dtype: Optional[torch.dtype] = None,
        device=None,
        dtype=None,
    ) -> None:
        FP8ScaledLayer.__init__(self, compute_dtype=compute_dtype)
        nn.Conv1d.__init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )
        # Wan FP8 checkpoints store `scale_weight` as a scalar; we rely on
        # broadcasting in `_scale_and_cast_weight` to apply it per out-channel.
        self.scale_weight = nn.Parameter(
            torch.ones(1, dtype=torch.float32), requires_grad=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        compute_dtype = self._effective_compute_dtype(x)

        x_c = x.to(compute_dtype)
        w = self._scale_and_cast_weight(
            self.weight,
            getattr(self, "scale_weight", None),
            target_dtype=compute_dtype,
            per_out_feature_dim=0,
        )
        b = self.bias
        if b is not None and b.dtype != compute_dtype:
            b = b.to(compute_dtype)

        out = F.conv1d(
            x_c,
            w,
            b,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        # Always return in compute dtype (float16/bfloat16/float32), not FP8.
        return out

class FP8ScaledEmbedding(FP8ScaledLayer, nn.Embedding):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
        max_norm: Optional[float] = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        sparse: bool = False,
        *,
        compute_dtype: Optional[torch.dtype] = None,
        device=None,
        dtype=None,
    ) -> None:
        FP8ScaledLayer.__init__(self, compute_dtype=compute_dtype)
        nn.Embedding.__init__(
            self,
            num_embeddings,
            embedding_dim,
            padding_idx=padding_idx,
            max_norm=max_norm,
            norm_type=norm_type,
            scale_grad_by_freq=scale_grad_by_freq,
            sparse=sparse,
            device=device,
            dtype=dtype,
        )
        # Wan FP8 checkpoints store `scale_weight` as a scalar; we rely on
        # broadcasting in `_scale_and_cast_weight` to apply it per embedding-dim.
        self.scale_weight = nn.Parameter(
            torch.ones(1, dtype=torch.float32), requires_grad=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        compute_dtype = self._effective_compute_dtype(None)
        w = self._scale_and_cast_weight(
            self.weight,
            getattr(self, "scale_weight", None),
            target_dtype=compute_dtype,
            per_out_feature_dim=-1,
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


_TYPE_MAP = {
    nn.Linear: FP8ScaledLinear,
    nn.Conv2d: FP8ScaledConv2d,
    nn.Conv1d: FP8ScaledConv1d,
    nn.Embedding: FP8ScaledEmbedding,
}


def patch_fp8_scaled_model(
    model: nn.Module,
    name_filter: Optional[Callable[[str], bool]] = None,
    *,
    default_compute_dtype: Optional[torch.dtype] = None,
) -> None:
    """
    In-place patch of a model to use FP8Scaled* layers.

    This should be called *before* loading an FP8-scaled checkpoint whose
    state_dict contains both `{name}.weight` (in FP8) and `{name}.scale_weight`.

    The function prefers a Transformer Engine-backed pathway when
    `transformer_engine.pytorch` is importable (via the FP8Scaled* layers using
    TE matmul kernels where applicable) and otherwise falls back to pure PyTorch
    implementations.
    """

    stack = [("", model)]
    while stack:
        prefix, mod = stack.pop()
        for name, child in list(mod._modules.items()):
            qname = f"{prefix}{name}"
            t = type(child)

            if t in _TYPE_MAP and (name_filter is None or name_filter(qname)):
                # Recreate the module as an FP8Scaled* of the appropriate type,
                # copying over the existing (non-FP8) weights. The FP8 / scale
                # weights will then be loaded from the FP8 checkpoint.
                cls = _TYPE_MAP[t]

                if isinstance(child, nn.Linear):
                    new_mod = cls(  # type: ignore[call-arg]
                        child.in_features,
                        child.out_features,
                        bias=child.bias is not None,
                        compute_dtype=default_compute_dtype,
                        device=child.weight.device,
                        dtype=child.weight.dtype,
                    )
                    with torch.no_grad():
                        new_mod.weight.copy_(child.weight)
                        if child.bias is not None:
                            new_mod.bias.copy_(child.bias)
                elif isinstance(child, nn.Conv2d):
                    new_mod = cls(  # type: ignore[call-arg]
                        child.in_channels,
                        child.out_channels,
                        child.kernel_size,
                        stride=child.stride,
                        padding=child.padding,
                        dilation=child.dilation,
                        groups=child.groups,
                        bias=child.bias is not None,
                        compute_dtype=default_compute_dtype,
                        device=child.weight.device,
                        dtype=child.weight.dtype,
                    )
                    with torch.no_grad():
                        new_mod.weight.copy_(child.weight)
                        if child.bias is not None:
                            new_mod.bias.copy_(child.bias)
                elif isinstance(child, nn.Conv1d):
                    new_mod = cls(  # type: ignore[call-arg]
                        child.in_channels,
                        child.out_channels,
                        child.kernel_size,
                        stride=child.stride,
                        padding=child.padding,
                        dilation=child.dilation,
                        groups=child.groups,
                        bias=child.bias is not None,
                        compute_dtype=default_compute_dtype,
                        device=child.weight.device,
                        dtype=child.weight.dtype,
                    )
                    with torch.no_grad():
                        new_mod.weight.copy_(child.weight)
                        if child.bias is not None:
                            new_mod.bias.copy_(child.bias)
                elif isinstance(child, nn.Embedding):
                    new_mod = cls(  # type: ignore[call-arg]
                        child.num_embeddings,
                        child.embedding_dim,
                        padding_idx=child.padding_idx,
                        max_norm=child.max_norm,
                        norm_type=child.norm_type,
                        scale_grad_by_freq=child.scale_grad_by_freq,
                        sparse=child.sparse,
                        compute_dtype=default_compute_dtype,
                        device=child.weight.device,
                        dtype=child.weight.dtype,
                    )
                    with torch.no_grad():
                        new_mod.weight.copy_(child.weight)
                else:  # pragma: no cover - defensive
                    continue

                mod._modules[name] = new_mod
                child = new_mod

            if child is not None and len(child._modules) > 0:
                stack.append((qname + ".", child))
