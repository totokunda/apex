from typing import Optional
import torch.nn as nn
import torch
import torch.nn.functional as F

from einops import rearrange
from src.attention import attention_register


class StepVideoAttnProcessor:
    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "StepVideoAttnProcessor requires PyTorch 2.0. To use it, please upgrade PyTorch to 2.0."
            )

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rope_positions: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        **kwargs,
    ) -> torch.Tensor:
        
        if hasattr(attn, "wqkv"):
            xqkv = attn.wqkv(hidden_states)
            xqkv = xqkv.view(*hidden_states.shape[:-1], attn.n_heads, 3 * attn.head_dim)

            xq, xk, xv = torch.split(
                xqkv, [attn.head_dim] * 3, dim=-1
            )  ## seq_len, n, dim

            if attn.with_qk_norm:
                xq = attn.q_norm(xq)
                xk = attn.k_norm(xk)

            if attn.with_rope:
                xq = attn.apply_rope3d(
                    xq, rope_positions, attn.rope_ch_split, parallel=attn.parallel
                )
                xk = attn.apply_rope3d(
                    xk, rope_positions, attn.rope_ch_split, parallel=attn.parallel
                )

        elif hasattr(attn, "wq"):
            xq = attn.wq(hidden_states)
            xq = xq.view(*xq.shape[:-1], attn.n_heads, attn.head_dim)

            xkv = attn.wkv(encoder_hidden_states)
            xkv = xkv.view(*xkv.shape[:-1], attn.n_heads, 2 * attn.head_dim)

            xk, xv = torch.split(xkv, [attn.head_dim] * 2, dim=-1)  ## seq_len, n, dim

            if attn.with_qk_norm:
                xq = attn.q_norm(xq)
                xk = attn.k_norm(xk)
        else:
            raise ValueError(f"Unsupported attention type: {type(attn)}")

        if attention_mask is not None and attention_mask.dtype != torch.bool:
            attention_mask = attention_mask.to(xq.dtype)

        if attention_mask is not None and attention_mask.ndim == 3:
            n_heads = xq.shape[2]
            attention_mask = attention_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)
        
        xq, xk, xv = map(lambda x: rearrange(x, 'b s h d -> b h s d'), (xq, xk, xv))

        output = attention_register.call(
            xq,
            xk,
            xv,
            attn_mask=attention_mask,
            key="sdpa"
        )
        output = rearrange(output, 'b h s d -> b s h d')
        output = rearrange(output, "b s h d -> b s (h d)")
        output = attn.wo(output)
        return output
