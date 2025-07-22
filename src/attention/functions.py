import torch
from src.register import FunctionRegister
import math
import warnings

try:
    from flash_attn_interface import flash_attn_func as flash_attn_func_3
except ImportError:
    flash_attn_func_3 = None

try:
    from flash_attn import flash_attn_varlen_func, flash_attn_func
except ImportError:
    flash_attn_func = None
    flash_attn_varlen_func = None

try:
    from torch_xla.experimental.custom_kernel import (
        flash_attention as xla_flash_attention_func,
    )
except ImportError:
    xla_flash_attention_func = None
    pass
try:
    from sageattention import sageattn
except ImportError:
    sageattn = None

try:
    from torch.nn.attention.flex_attention import flex_attention

    flex_attention = torch.compile(flex_attention)
except ImportError:
    flex_attention = None


attention_register = FunctionRegister()


@attention_register("sdpa")
def sdpa_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, **kwargs):
    return torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal
    ).transpose(1, 2)


def flash_attention_padded(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    softmax_scale: float | None = None,
    default_dtype: torch.dtype = torch.bfloat16,
    is_causal: bool = False,
    **kwargs,
) -> torch.Tensor:
    """
    FlashAttention-2 **padded** (non-var-len) wrapper that works for *self-* **and**
    *cross-attention*.

    It accepts `(B, H, S, D)` tensors.  Sequence lengths may differ between
    `q` and `k/v` (cross-attention).  Internally everything runs in **bf16 or
    fp16 only** – never float32.

    Parameters
    ----------
    q : (B, H, Sq, D) tensor
    k : (B, H, Sk, D) tensor
    v : (B, H, Sk, D) tensor
        *H* (num heads) and *D* (head dimension) must match across all three.
    softmax_scale : float, optional
        Defaults to ``1/sqrt(D)``.
    default_dtype : torch.dtype, default ``torch.bfloat16``
        Used when an input arrives in an unsupported dtype.
    is_causal : bool, default ``False``
        Apply a causal mask (only makes sense for self-attention).

    Returns
    -------
    out : (B, H, Sq, D) tensor
    """
    # ------------------------------------------------------------------ #
    if flash_attn_func is None:
        raise ImportError(
            "flash_attn is not installed or flash_attn_func missing. "
            "Install FlashAttention-2 (pip install flash-attn)."
        )

    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k, v must have shape (B, H, S, D)")

    Bq, Hq, Sq, Dq = q.shape
    Bk, Hk, Sk, Dk = k.shape
    Bv, Hv, Sv, Dv = v.shape

    if not (Bq == Bk == Bv):
        raise ValueError("Batch sizes differ (q, k, v); pad or split batches first.")
    if not (Hq == Hk == Hv and Dq == Dk == Dv):
        raise ValueError("Head counts / head dims mismatch across q, k, v.")
    if not (Sk == Sv):
        raise ValueError("Key and value sequence lengths must match (Sk == Sv).")

    # ------------------------------------------------------------------ #
    # Make sure we are using a supported low-precision dtype
    #
    allowed = {torch.bfloat16, torch.float16}
    for name, t in (("q", q), ("k", k), ("v", v)):
        if t.dtype not in allowed:
            warnings.warn(
                f"{name} has dtype {t.dtype} – casting to {default_dtype}.",
                stacklevel=2,
            )

    q = q.to(default_dtype if q.dtype not in allowed else q.dtype).contiguous()
    k = k.to(default_dtype if k.dtype not in allowed else k.dtype).contiguous()
    v = v.to(default_dtype if v.dtype not in allowed else v.dtype).contiguous()

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(Dq)

    # ------------------------------------------------------------------ #
    # FlashAttention kernel needs (B, Sq, H, D) layout
    #
    q_in = q.permute(0, 2, 1, 3)  # (B, Sq, H, D)
    k_in = k.permute(0, 2, 1, 3)  # (B, Sk, H, D)
    v_in = v.permute(0, 2, 1, 3)  # (B, Sk, H, D)

    out = flash_attn_func(
        q_in,
        k_in,
        v_in,
        causal=is_causal,
        softmax_scale=softmax_scale,
        dropout_p=0.0,  # change if training-time dropout desired
    )

    # Back to (B, H, Sq, D) and caller’s dtype
    return out.permute(0, 2, 1, 3).to(q.dtype).transpose(1, 2)


def flash_attention_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    cu_seqlens_q: torch.Tensor | None = None,
    cu_seqlens_k: torch.Tensor | None = None,
    max_seqlen_q: int | None = None,
    max_seqlen_k: int | None = None,
    softmax_scale: float | None = None,
    default_dtype: torch.dtype = torch.bfloat16,
    is_causal: bool = False,
    **kwargs,
) -> torch.Tensor:
    """
    FlashAttention-2 var-len wrapper that supports both *self* and *cross* attention.

    Parameters
    ----------
    q : (Bq, H, Sq, D) tensor
    k : (Bk, H, Sk, D) tensor
    v : (Bk, H, Sk, D) tensor
        *H* (num heads) and *D* (head dim) must match across all three tensors.
        Usually `Bq == Bk`, but the wrapper only assumes `Bk >= Bq`
        (common case in encoder–decoder cross-attn with packed memory).
    cu_seqlens_q, cu_seqlens_k : (batch+1,) int32 tensors, optional
        Cumulative sequence-length vectors:
          `[0, len₀, len₀+len₁, …]`.
        If **omitted** we assume *uniform* lengths and build them automatically.
    max_seqlen_q, max_seqlen_k : int, optional
        Maximum sequence length for q / k-v.  Inferred if not given.
    softmax_scale : float, optional
        Defaults to `1/√D` if `None`.
    default_dtype : torch.dtype, default **bfloat16**
        Used when any input arrives in an unsupported dtype.
    is_causal : bool, default **False**
        Apply a causal mask (useful for decoder self-attention).

    Returns
    -------
    out : (Bq, H, Sq, D) tensor
    """
    # -------------------- checks & dtype sanitisation -------------------- #
    if flash_attn_varlen_func is None:
        raise ImportError(
            "flash_attn is not installed or flash_attn_varlen_func is undefined."
        )

    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k, v must have shape (B, H, S, D)")

    Bq, Hq, Sq, Dq = q.shape
    Bk, Hk, Sk, Dk = k.shape
    Bv, Hv, Sv, Dv = v.shape

    if not (Hq == Hk == Hv and Dq == Dk == Dv and Sk == Sv):
        raise ValueError("Mismatched head counts / head dims or K ≠ V shapes")

    accepted = {torch.bfloat16, torch.float16}
    for name, t in (("q", q), ("k", k), ("v", v)):
        if t.dtype not in accepted:
            warnings.warn(
                f"{name} is {t.dtype}. Casting to {default_dtype} (never float32).",
                stacklevel=2,
            )

    q = q.to(default_dtype if q.dtype not in accepted else q.dtype).contiguous()
    k = k.to(default_dtype if k.dtype not in accepted else k.dtype).contiguous()
    v = v.to(default_dtype if v.dtype not in accepted else v.dtype).contiguous()

    # ------------------ build (or validate) cu_seqlens ------------------- #
    device = q.device
    if cu_seqlens_q is None:
        cu_seqlens_q = torch.arange(
            0, (Bq + 1) * Sq, Sq, dtype=torch.int32, device=device
        )
    if cu_seqlens_k is None:
        cu_seqlens_k = torch.arange(
            0, (Bk + 1) * Sk, Sk, dtype=torch.int32, device=device
        )

    if cu_seqlens_q.dtype != torch.int32 or cu_seqlens_k.dtype != torch.int32:
        raise TypeError("cu_seqlens tensors must be int32")

    if max_seqlen_q is None:
        max_seqlen_q = int((cu_seqlens_q[1:] - cu_seqlens_q[:-1]).max().item())
    if max_seqlen_k is None:
        max_seqlen_k = int((cu_seqlens_k[1:] - cu_seqlens_k[:-1]).max().item())

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(Dq)

    # --------------------- flatten to (T, H, D) -------------------------- #
    # FlashAttention-2 expects contiguous `(total_tokens, num_heads, head_dim)`
    #
    q_flat = q.permute(0, 2, 1, 3).reshape(-1, Hq, Dq)
    k_flat = k.permute(0, 2, 1, 3).reshape(-1, Hq, Dq)
    v_flat = v.permute(0, 2, 1, 3).reshape(-1, Hq, Dq)

    # ----------------------- kernel invocation --------------------------- #
    out_flat = flash_attn_varlen_func(
        q=q_flat,
        k=k_flat,
        v=v_flat,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        dropout_p=0.0,  # change if training-time dropout desired
        softmax_scale=softmax_scale,
        causal=is_causal,
    )

    # --------------------------- re-shape -------------------------------- #
    out = (
        out_flat.reshape(Bq, Sq, Hq, Dq)
        .permute(0, 2, 1, 3)  # (Bq, H, Sq, D)
        .to(q.dtype)  # preserve caller’s dtype
    )
    return out.transpose(1, 2)


@attention_register("flash")
def flash_attention(
    q,
    k,
    v,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    max_seqlen_q=None,
    max_seqlen_k=None,
    softmax_scale=None,
    default_dtype=torch.bfloat16,
    is_causal=False,
    **kwargs,
):
    if cu_seqlens_q is None or cu_seqlens_k is None:
        return flash_attention_padded(
            q,
            k,
            v,
            softmax_scale=softmax_scale,
            default_dtype=default_dtype,
            is_causal=is_causal,
            **kwargs,
        )
    else:
        return flash_attention_varlen(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=softmax_scale,
            default_dtype=default_dtype,
            is_causal=is_causal,
            **kwargs,
        )


@attention_register("flash3")
def flash_attention3(
    q, k, v, softmax_scale=None, default_dtype=torch.bfloat16, is_causal=False, **kwargs
):
    if flash_attn_func_3 is None:
        raise ImportError("flash_attn_interface is not installed")

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    start_dtype = q.dtype
    acceptable_dtypes = [torch.bfloat16, torch.float16]
    if q.dtype not in acceptable_dtypes:
        q = q.to(default_dtype)
    if k.dtype not in acceptable_dtypes:
        k = k.to(default_dtype)
    if v.dtype not in acceptable_dtypes:
        v = v.to(default_dtype)

    out = flash_attn_func_3(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        softmax_scale=softmax_scale,
        causal=is_causal,
    )

    # check if out is a tuple of two tensors
    if isinstance(out, tuple):
        return out[0]
    else:
        return out


@attention_register("sage")
def sage_attention(q, k, v, is_causal=False, **kwargs):
    attn_output = sageattn(q, k, v, tensor_layout="HND", is_causal=is_causal)
    return attn_output.transpose(1, 2)


@attention_register("xla_flash")
def xla_flash_attention(q, k, v, attention_mask, softmax_scale, **kwargs):
    batch_size = q.shape[0]
    q_segment_indexes = None
    if (
        attention_mask is not None
    ):  # if mask is required need to tune both segmenIds fields
        # attention_mask = torch.squeeze(attention_mask).to(torch.float32)
        attention_mask = attention_mask.to(torch.float32)
        q_segment_indexes = torch.ones(
            batch_size, q.shape[2], device=q.device, dtype=torch.float32
        )
        assert (
            attention_mask.shape[1] == k.shape[2]
        ), f"ERROR: KEY SHAPE must be same as attention mask [{k.shape[2]}, {attention_mask.shape[1]}]"
    assert (
        q.shape[2] % 128 == 0
    ), f"ERROR: QUERY SHAPE must be divisible by 128 (TPU limitation) [{q.shape[2]}]"

    assert (
        k.shape[2] % 128 == 0
    ), f"ERROR: KEY SHAPE must be divisible by 128 (TPU limitation) [{k.shape[2]}]"

    if xla_flash_attention_func is None:
        raise ImportError("xla_flash_attention is not installed")
    return xla_flash_attention_func(
        q, k, v, q_segment_indexes, attention_mask, softmax_scale
    )


@attention_register("flex")
def flex_attention_func(q, k, v, attn_mask=None, softmax_scale=None, **kwargs):
    if flex_attention is None:
        raise ImportError("flex_attention is not installed")
    return flex_attention(q, k, v, block_mask=attn_mask, scale=softmax_scale, **kwargs)
