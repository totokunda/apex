# Copyright (c) 2025 SandAI. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin
from diffusers.utils import (
    USE_PEFT_BACKEND,
    logging,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models.attention_processor import Attention
from diffusers.models.cache_utils import CacheMixin
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin

from src.attention import attention_register
from src.attention.processors.magi_processor import MagiAttentionProcessor, MagiCrossAttentionProcessor
from src.transformer_models.base import TRANSFORMERS_REGISTRY

logger = logging.get_logger(__name__)


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size: int, cond_hidden_ratio: float = 1.0, frequency_embedding_size: int = 256):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, int(hidden_size * cond_hidden_ratio), bias=True),
            nn.SiLU(),
            nn.Linear(
                int(hidden_size * cond_hidden_ratio), int(hidden_size * cond_hidden_ratio), bias=True
            ),
        )
        self.frequency_embedding_size = frequency_embedding_size
        self.timestep_rescale_factor = 1000

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000, timestep_rescale_factor=1):
        """
        Create sinusoidal timestep embeddings.
        """
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
            device=t.device
        )
        args = t[:, None].float() * freqs[None] * timestep_rescale_factor
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t = t.to(torch.float32)
        t_freq = self.timestep_embedding(
            t, self.frequency_embedding_size, timestep_rescale_factor=self.timestep_rescale_factor
        )
        t_emb = self.mlp(t_freq)
        return t_emb


class CaptionEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """

    def __init__(self, caption_channels: int, hidden_size: int, xattn_cond_hidden_ratio: float = 1.0, 
                 cond_hidden_ratio: float = 1.0, caption_max_length: int = 256):
        super().__init__()

        self.y_proj_xattn = nn.Sequential(
            nn.Linear(caption_channels, int(hidden_size * xattn_cond_hidden_ratio), bias=True), nn.SiLU()
        )

        self.y_proj_adaln = nn.Sequential(nn.Linear(caption_channels, int(hidden_size * cond_hidden_ratio), bias=True))

        self.null_caption_embedding = nn.Parameter(torch.empty(caption_max_length, caption_channels))

    def caption_drop(self, caption, caption_dropout_mask):
        """
        Drops labels to enable classifier-free guidance.
        """
        dropped_caption = torch.where(
            caption_dropout_mask[:, None, None, None],
            self.null_caption_embedding[None, None, :],
            caption,
        )
        return dropped_caption

    def caption_drop_single_token(self, caption_dropout_mask):
        dropped_caption = torch.where(
            caption_dropout_mask[:, None, None],
            self.null_caption_embedding[None, -1, :],
            self.null_caption_embedding[None, -2, :],
        )
        return dropped_caption

    def forward(self, caption, train=False, caption_dropout_mask=None):
        if train and caption_dropout_mask is not None:
            caption = self.caption_drop(caption, caption_dropout_mask)
        caption_xattn = self.y_proj_xattn(caption)
        if caption_dropout_mask is not None:
            caption = self.caption_drop_single_token(caption_dropout_mask)

        caption_adaln = self.y_proj_adaln(caption)
        return caption_xattn, caption_adaln


class LearnableRotaryEmbeddingCat(nn.Module):
    """Rotary position embedding w/ concatenated sin & cos"""

    def __init__(self, dim, max_res=224, temperature=10000, in_pixels=True, linear_bands: bool = False):
        super().__init__()
        self.dim = dim
        self.max_res = max_res
        self.temperature = temperature
        self.in_pixels = in_pixels
        self.linear_bands = linear_bands
        self.bands = nn.Parameter(self.get_default_bands())

    def get_default_bands(self):
        if self.in_pixels:
            bands = self.pixel_freq_bands(
                self.dim // 8, float(self.max_res), linear_bands=self.linear_bands
            )
        else:
            bands = self.freq_bands(self.dim // 8, temperature=self.temperature, step=1)
        return bands

    def pixel_freq_bands(self, num_bands: int, max_freq: float = 224.0, linear_bands: bool = True):
        if linear_bands:
            bands = torch.linspace(1.0, max_freq / 2, num_bands, dtype=torch.float32)
        else:
            bands = 2 ** torch.linspace(0, math.log(max_freq, 2) - 1, num_bands, dtype=torch.float32)
        return bands * torch.pi

    def freq_bands(self, num_bands: int, temperature: float = 10000.0, step: int = 2):
        exp = torch.arange(0, num_bands, step, dtype=torch.int64).to(torch.float32) / num_bands
        bands = 1.0 / (temperature**exp)
        return bands

    def get_embed(self, shape, ref_feat_shape=None):
        # Build rotary position embeddings
        if self.in_pixels:
            t = [torch.linspace(-1.0, 1.0, steps=s, dtype=torch.float32) for s in shape]
        else:
            t = [torch.arange(s, dtype=torch.int64).to(torch.float32) for s in shape]
            t[1] = t[1] - (shape[1] - 1) / 2
            t[2] = t[2] - (shape[2] - 1) / 2
        
        if ref_feat_shape is not None:
            t_rescaled = []
            for x, f, r in zip(t, shape, ref_feat_shape):
                if f == 1:
                    t_rescaled.append(x)
                else:
                    t_rescaled.append(x / (f - 1) * (r - 1))
            t = t_rescaled

        grid = torch.stack(torch.meshgrid(t, indexing="ij"), dim=-1)
        grid = grid.unsqueeze(-1)
        pos = grid * self.bands

        pos_sin, pos_cos = pos.sin(), pos.cos()
        num_spatial_dim = math.prod(shape)
        sin_emb = pos_sin.reshape(num_spatial_dim, -1)
        cos_emb = pos_cos.reshape(num_spatial_dim, -1)
        return torch.cat([sin_emb, cos_emb], -1)


class FinalLinear(nn.Module):
    """
    The final linear layer of DiT.
    """

    def __init__(self, hidden_size, patch_size, t_patch_size, out_channels):
        super().__init__()
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * t_patch_size * out_channels, bias=False)

    def forward(self, x):
        x = self.linear(x)
        return x


class AdaModulateLayer(torch.nn.Module):
    def __init__(self, hidden_size: int, cond_hidden_ratio: float = 1.0, cond_gating_ratio: float = 1.0):
        super().__init__()
        self.gate_num_chunks = 2
        self.act = nn.SiLU()
        self.proj = nn.Sequential(
            nn.Linear(
                int(hidden_size * cond_hidden_ratio),
                int(hidden_size * cond_gating_ratio * self.gate_num_chunks),
                bias=True,
            )
        )

    def forward(self, c):
        c = self.act(c)
        return self.proj(c)


@maybe_allow_in_graph
class MagiTransformerBlock(nn.Module):
    """
    Transformer block used in MAGI model.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        cross_attention_dim: int,
        activation_fn: str = "gelu-approximate",
        attention_bias: bool = True,
        attention_out_bias: bool = True,
        eps: float = 1e-6,
        elementwise_affine: bool = False,
        cond_hidden_ratio: float = 1.0,
        cond_gating_ratio: float = 1.0,
    ):
        super().__init__()

        self.ada_modulate_layer = AdaModulateLayer(
            hidden_size=dim, 
            cond_hidden_ratio=cond_hidden_ratio, 
            cond_gating_ratio=cond_gating_ratio
        )

        # Self attention
        self.norm1 = nn.LayerNorm(dim, eps=eps, elementwise_affine=elementwise_affine)
        self.attn1 = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            bias=attention_bias,
            cross_attention_dim=None,
            out_bias=attention_out_bias,
            processor=MagiAttentionProcessor(),
        )

        # Cross attention
        self.norm2 = nn.LayerNorm(dim, eps=eps, elementwise_affine=elementwise_affine)
        self.attn2 = Attention(
            query_dim=dim,
            cross_attention_dim=cross_attention_dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            bias=attention_bias,
            out_bias=attention_out_bias,
            processor=MagiCrossAttentionProcessor(),
        )

        # MLP
        self.norm3 = nn.LayerNorm(dim, eps=eps, elementwise_affine=elementwise_affine)
        mlp_hidden_dim = int(dim * 4.0)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim, bias=False),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, dim, bias=False),
        )

        # Post norms
        self.self_attn_post_norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=elementwise_affine)
        self.mlp_post_norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=elementwise_affine)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        condition: torch.Tensor,
        condition_map: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Get gating values from condition based on condition_map
        # Use condition_map to select the right condition for each token
        batch_size = condition.shape[0]
        seq_len = hidden_states.shape[0]
        
        # Handle condition mapping properly
        if condition_map.numel() == seq_len:
            # Simple case: one condition per token
            selected_conditions = condition.view(-1, condition.shape[-1])[condition_map.long()]
        else:
            # More complex case: need to handle chunking
            selected_conditions = condition.view(-1, condition.shape[-1])[:seq_len]

        gate_output = self.ada_modulate_layer(selected_conditions)
        gate_output = torch.tanh(gate_output)  # softcap as in original MAGI
        gate_msa, gate_mlp = gate_output.chunk(2, dim=-1)

        residual = hidden_states

        # Self attention with gating
        norm_hidden_states = self.norm1(hidden_states)
        
        # Use the attention processor which handles rotary embeddings
        attn_output = self.attn1(
            norm_hidden_states,
            rotary_pos_emb=rotary_pos_emb,
        )

        # Apply condition-based gating and residual (following MAGI's range_mod pattern)
        attn_output = attn_output * gate_msa
        attn_output = self.self_attn_post_norm(attn_output)
        hidden_states = residual + attn_output

        # Cross attention
        attn_output = self.attn2(
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=encoder_attention_mask,
        )
        hidden_states = hidden_states + attn_output

        residual = hidden_states
        
        # MLP with gating
        norm_hidden_states = self.norm3(hidden_states)
        mlp_output = self.mlp(norm_hidden_states)
        
        # Apply gating and post norm
        mlp_output = mlp_output * gate_mlp
        mlp_output = self.mlp_post_norm(mlp_output)
        hidden_states = residual + mlp_output

        return hidden_states


@maybe_allow_in_graph
@TRANSFORMERS_REGISTRY("magi")
class MagiTransformer3DModel(
    ModelMixin, ConfigMixin, FromOriginalModelMixin, PeftAdapterMixin, CacheMixin
):
    """
    A Transformer model for video-like data used in MAGI.
    This model provides both diffusers-compatible interface and MAGI-original interface.
    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        in_channels: int = 32,
        out_channels: int = 32,
        patch_size: int = 2,
        t_patch_size: int = 1,
        num_attention_heads: int = 32,
        attention_head_dim: int = 64,
        cross_attention_dim: int = 2048,
        num_layers: int = 28,
        activation_fn: str = "gelu-approximate",
        attention_bias: bool = True,
        attention_out_bias: bool = True,
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-6,
        caption_channels: int = 4096,
        caption_max_length: int = 256,
        cond_hidden_ratio: float = 1.0,
        xattn_cond_hidden_ratio: float = 1.0,
        cond_gating_ratio: float = 1.0,
        x_rescale_factor: float = 1.0,
        half_channel_vae: bool = False,
    ) -> None:
        super().__init__()

        out_channels = out_channels or in_channels
        inner_dim = num_attention_heads * attention_head_dim

        # Input projection
        self.x_embedder = nn.Conv3d(
            in_channels,
            inner_dim,
            kernel_size=(t_patch_size, patch_size, patch_size),
            stride=(t_patch_size, patch_size, patch_size),
            bias=False,
        )

        # Time embedding
        self.t_embedder = TimestepEmbedder(
            hidden_size=inner_dim, 
            cond_hidden_ratio=cond_hidden_ratio
        )

        # Caption embedding
        self.y_embedder = CaptionEmbedder(
            caption_channels=caption_channels,
            hidden_size=inner_dim,
            xattn_cond_hidden_ratio=xattn_cond_hidden_ratio,
            cond_hidden_ratio=cond_hidden_ratio,
            caption_max_length=caption_max_length,
        )

        # Rotary position embedding
        self.rope = LearnableRotaryEmbeddingCat(
            inner_dim // num_attention_heads, in_pixels=False
        )

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList(
            [
                MagiTransformerBlock(
                    dim=inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    cross_attention_dim=int(inner_dim * xattn_cond_hidden_ratio),
                    activation_fn=activation_fn,
                    attention_bias=attention_bias,
                    attention_out_bias=attention_out_bias,
                    eps=norm_eps,
                    elementwise_affine=norm_elementwise_affine,
                    cond_hidden_ratio=cond_hidden_ratio,
                    cond_gating_ratio=cond_gating_ratio,
                )
                for _ in range(num_layers)
            ]
        )

        # Output layers
        self.norm_out = nn.LayerNorm(inner_dim, eps=norm_eps, elementwise_affine=norm_elementwise_affine)
        self.final_linear = FinalLinear(inner_dim, patch_size, t_patch_size, out_channels)

        self.gradient_checkpointing = False

    def unpatchify(self, x, H, W):
        return rearrange(
            x,
            "(T H W) N (pT pH pW C) -> N C (T pT) (H pH) (W pW)",
            H=H,
            W=W,
            pT=self.config.t_patch_size,
            pH=self.config.patch_size,
            pW=self.config.patch_size,
        ).contiguous()

    def _prepare_embeddings_and_conditions(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        caption_dropout_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prepare embeddings and conditions for MAGI transformer blocks.
        This follows the MAGI preprocessing pattern but simplified for diffusers interface.
        """
        batch_size, _, T, H, W = hidden_states.shape

        # Apply rescaling if needed
        if self.config.x_rescale_factor != 1.0:
            hidden_states = hidden_states * self.config.x_rescale_factor

        if self.config.half_channel_vae:
            assert hidden_states.shape[1] == 16
            hidden_states = torch.cat([hidden_states, hidden_states], dim=1)

        # Embed input
        hidden_states = self.x_embedder(hidden_states)  # [N, C, T, H, W]
        
        # Rearrange to sequence format (following MAGI pattern)
        hidden_states = rearrange(hidden_states, "N C T H W -> (T H W) N C").contiguous()

        # Get rotary position embeddings (following MAGI pattern)
        rescale_factor = math.sqrt((H * W) / (16 * 16))
        rope = self.rope.get_embed(
            shape=[T, H, W], 
            ref_feat_shape=[T, H / rescale_factor, W / rescale_factor]
        )

        # Time embedding
        t_emb = self.t_embedder(timestep.flatten())
        
        # Caption embedding (following MAGI pattern)
        y_xattn, y_adaln = self.y_embedder(
            encoder_hidden_states, 
            train=self.training, 
            caption_dropout_mask=caption_dropout_mask
        )

        # Prepare condition (following MAGI's condition preparation)
        if y_adaln.dim() > 2:
            y_adaln = y_adaln.squeeze(1)
        
        # Handle different timestep shapes for chunked processing
        if t_emb.dim() == 1:
            t_emb = t_emb.unsqueeze(0)
        if t_emb.shape[0] != batch_size:
            t_emb = t_emb.repeat(batch_size, 1)
        
        condition = t_emb + y_adaln

        # Create condition map (simplified version of MAGI's complex condition mapping)
        seq_len = hidden_states.size(0)
        condition_map = torch.arange(batch_size, device=hidden_states.device)
        condition_map = condition_map.repeat_interleave(seq_len // batch_size)

        # Prepare cross attention inputs (following MAGI pattern)
        if encoder_attention_mask is not None:
            encoder_attention_mask = encoder_attention_mask.squeeze()
            y_xattn_flat = torch.masked_select(
                y_xattn.squeeze(1), 
                encoder_attention_mask.unsqueeze(-1).bool()
            ).reshape(-1, y_xattn.shape[-1])
        else:
            y_xattn_flat = y_xattn.squeeze(1).flatten(0, 1)

        return hidden_states, condition, condition_map, y_xattn_flat, rope

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_attention_mask: torch.Tensor = None,
        caption_dropout_mask: torch.Tensor = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
        # MAGI-specific parameters for compatibility
        num_frames: Optional[int] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        **kwargs,
    ) -> Union[torch.Tensor, Transformer2DModelOutput]:
        """
        Forward pass that provides both diffusers interface and MAGI compatibility.
        """
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            scale_lora_layers(self, lora_scale)

        # Store original shape for unpatchify
        _, _, T, H, W = hidden_states.shape

        # Prepare embeddings and conditions
        hidden_states, condition, condition_map, y_xattn_flat, rope = self._prepare_embeddings_and_conditions(
            hidden_states, timestep, encoder_hidden_states, caption_dropout_mask, encoder_attention_mask
        )

        # Apply transformer blocks
        for block in self.transformer_blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    y_xattn_flat,
                    condition,
                    condition_map,
                    rope,
                    encoder_attention_mask,
                )
            else:
                hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=y_xattn_flat,
                    condition=condition,
                    condition_map=condition_map,
                    rotary_pos_emb=rope,
                    encoder_attention_mask=encoder_attention_mask,
                )

        # Final processing
        hidden_states = self.norm_out(hidden_states)
        hidden_states = self.final_linear(hidden_states)

        # Unpatchify
        output = self.unpatchify(hidden_states, H, W)

        # Apply post-processing (following MAGI pattern)
        if self.config.half_channel_vae:
            assert output.shape[1] == 32
            output = output[:, :16]

        if self.config.x_rescale_factor != 1.0:
            output = output / self.config.x_rescale_factor

        if USE_PEFT_BACKEND:
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)

    def forward_magi_original(
        self,
        x: torch.Tensor,
        t: torch.Tensor, 
        y: torch.Tensor,
        caption_dropout_mask: Optional[torch.Tensor] = None,
        xattn_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Forward method that matches the original MAGI DiT interface.
        This is used for compatibility with the original MAGI inference system.
        """
        # Convert MAGI parameters to diffusers format
        hidden_states = x
        timestep = t.flatten() if t.dim() > 1 else t
        encoder_hidden_states = y
        encoder_attention_mask = xattn_mask
        
        # Call the main forward method
        result = self.forward(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            encoder_attention_mask=encoder_attention_mask,
            caption_dropout_mask=caption_dropout_mask,
            return_dict=False,
            **kwargs
        )
        
        return result[0]
