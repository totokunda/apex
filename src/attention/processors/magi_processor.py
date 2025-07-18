from typing import Optional

import torch
import torch.nn.functional as F
from diffusers.models.attention import Attention

from src.attention.functions import attention_register


def apply_rotary_emb(x, cos_emb, sin_emb):
    """Apply rotary position embeddings"""
    def rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    
    return x * cos_emb + rotate_half(x) * sin_emb


class MagiAttentionProcessor:
    r"""
    Processor for implementing attention in MAGI model with rotary embeddings and condition-based gating.
    """

    def __init__(self):
        pass

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        batch_size, sequence_length, _ = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(
                attention_mask, sequence_length, batch_size
            )
            attention_mask = attention_mask.view(
                batch_size, attn.heads, -1, attention_mask.shape[-1]
            )

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        # Apply normalization if available
        if hasattr(attn, 'norm_q') and attn.norm_q is not None:
            query = attn.norm_q(query)
        if hasattr(attn, 'norm_k') and attn.norm_k is not None:
            key = attn.norm_k(key)

        # Reshape to multi-head format
        query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

        # Apply rotary embeddings if provided
        if rotary_pos_emb is not None:
            sin_emb, cos_emb = rotary_pos_emb.tensor_split(2, -1)
            # Expand to match query/key dimensions
            cos_emb = cos_emb.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, head_dim)
            sin_emb = sin_emb.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, head_dim)
            
            query = apply_rotary_emb(query, cos_emb, sin_emb)
            key = apply_rotary_emb(key, cos_emb, sin_emb)

        # Use registered attention function
        hidden_states = attention_register.call(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)

        # Apply output projection
        hidden_states = attn.to_out[0](hidden_states)
        if len(attn.to_out) > 1:
            hidden_states = attn.to_out[1](hidden_states)
        
        return hidden_states


class MagiCrossAttentionProcessor:
    r"""
    Processor for implementing cross-attention in MAGI model.
    """

    def __init__(self):
        pass

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        batch_size = hidden_states.shape[0]
        sequence_length = hidden_states.shape[1]
        
        if encoder_hidden_states is not None:
            encoder_sequence_length = encoder_hidden_states.shape[1]
        else:
            encoder_sequence_length = sequence_length
            encoder_hidden_states = hidden_states

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(
                attention_mask, encoder_sequence_length, batch_size
            )
            attention_mask = attention_mask.view(
                batch_size, attn.heads, -1, attention_mask.shape[-1]
            )

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        # Apply normalization if available
        if hasattr(attn, 'norm_q') and attn.norm_q is not None:
            query = attn.norm_q(query)
        if hasattr(attn, 'norm_k') and attn.norm_k is not None:
            key = attn.norm_k(key)

        # Reshape to multi-head format
        query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

        # Use registered attention function
        hidden_states = attention_register.call(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)

        # Apply output projection
        hidden_states = attn.to_out[0](hidden_states)
        if len(attn.to_out) > 1:
            hidden_states = attn.to_out[1](hidden_states)
        
        return hidden_states
