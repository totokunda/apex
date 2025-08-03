from typing import Optional
import torch
import torch.nn.functional as F
from diffusers.models.attention_processor import Attention
from src.attention import attention_register


class Magi1AttnProcessor2_0:
    r"""
    Processor for implementing MAGI-1 attention mechanism.

    This processor handles both self-attention and cross-attention for the MAGI-1 model, following diffusers' standard
    attention processor interface. It supports image conditioning for image-to-video generation tasks.
    """

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "Magi1AttnProcessor2_0 requires PyTorch 2.0. To use it, please upgrade PyTorch to 2.0."
            )

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Handle image conditioning if present for cross-attention
        encoder_hidden_states_img = None
        if attn.add_k_proj is not None and encoder_hidden_states is not None:
            # Extract image conditioning from the concatenated encoder states
            # The text encoder context length is typically 512 tokens
            text_context_length = getattr(attn, "text_context_length", 512)
            image_context_length = encoder_hidden_states.shape[1] - text_context_length
            encoder_hidden_states_img = encoder_hidden_states[:, :image_context_length]
            encoder_hidden_states = encoder_hidden_states[:, image_context_length:]

        # For self-attention, use hidden_states as encoder_hidden_states
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        # Standard attention computation
        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        # Apply normalization if available
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Reshape for multi-head attention
        query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

        # Apply rotary embeddings if provided
        if rotary_emb is not None:

            def apply_rotary_emb(hidden_states: torch.Tensor, freqs: torch.Tensor):
                dtype = (
                    torch.float32
                    if hidden_states.device.type == "mps"
                    else torch.float64
                )
                x_rotated = torch.view_as_complex(
                    hidden_states.to(dtype).unflatten(3, (-1, 2))
                )
                x_out = torch.view_as_real(x_rotated * freqs).flatten(3, 4)
                return x_out.type_as(hidden_states)

            query = apply_rotary_emb(query, rotary_emb)
            key = apply_rotary_emb(key, rotary_emb)

        # Compute attention
        hidden_states = attention_register.call(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)

        # Handle image conditioning (I2V task) for cross-attention
        if encoder_hidden_states_img is not None:
            key_img = attn.add_k_proj(encoder_hidden_states_img)
            key_img = attn.norm_added_k(key_img)
            value_img = attn.add_v_proj(encoder_hidden_states_img)

            key_img = key_img.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            value_img = value_img.unflatten(2, (attn.heads, -1)).transpose(1, 2)

            attn_output_img = attention_register.call(
                query,
                key_img,
                value_img,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
            )
            attn_output_img = attn_output_img.transpose(1, 2).flatten(2, 3)
            hidden_states = hidden_states + attn_output_img

        # Apply output projection
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states
