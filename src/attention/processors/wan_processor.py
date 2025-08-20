import torch
import torch.nn.functional as F
from diffusers.models.attention import Attention
from typing import Optional
from einops import rearrange
import torch.nn as nn
from src.attention.functions import attention_register


NUM_FRAMES = None
ENHANCE_WEIGHT = None
ENABLE_ENHANCE = False


def get_num_frames() -> int:
    return NUM_FRAMES


def get_enhance_weight() -> float:
    return ENHANCE_WEIGHT


def enhance_score(query_image, key_image, head_dim, num_frames):
    scale = head_dim**-0.5
    query_image = query_image * scale
    attn_temp = query_image @ key_image.transpose(-2, -1)  # translate attn to float32
    attn_temp = attn_temp.to(torch.float32)
    attn_temp = attn_temp.softmax(dim=-1)

    # Reshape to [batch_size * num_tokens, num_frames, num_frames]
    attn_temp = attn_temp.reshape(-1, num_frames, num_frames)

    # Create a mask for diagonal elements
    diag_mask = torch.eye(num_frames, device=attn_temp.device).bool()
    diag_mask = diag_mask.unsqueeze(0).expand(attn_temp.shape[0], -1, -1)

    # Zero out diagonal elements
    attn_wo_diag = attn_temp.masked_fill(diag_mask, 0)

    # Calculate mean for each token's attention matrix
    # Number of off-diagonal elements per matrix is n*n - n
    num_off_diag = num_frames * num_frames - num_frames
    mean_scores = attn_wo_diag.sum(dim=(1, 2)) / num_off_diag

    enhance_scores = mean_scores.mean() * (num_frames + get_enhance_weight())
    enhance_scores = enhance_scores.clamp(min=1)
    return enhance_scores


def apply_rotary_emb(hidden_states: torch.Tensor, freqs: torch.Tensor):
    dtype = torch.float32 if hidden_states.device.type == "mps" else torch.float64
    x_rotated = torch.view_as_complex(hidden_states.to(dtype).unflatten(3, (-1, 2)))
    x_out = torch.view_as_real(x_rotated * freqs).flatten(3, 4)
    return x_out.type_as(hidden_states)

def rope_apply_ip_adapter(x, freqs, num_heads):
    x = rearrange(x, "b s (n d) -> b s n d", n=num_heads)
    x_out = torch.view_as_complex(
        x.to(torch.float64).reshape(x.shape[0], x.shape[1], x.shape[2], -1, 2)
    )
    x_out = torch.view_as_real(x_out * freqs).flatten(2)
    return x_out.to(x.dtype)



class WanAttnProcessor2_0:
    def __init__(self, use_enhance: bool = False):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "WanAttnProcessor2_0 requires PyTorch 2.0. To use it, please upgrade PyTorch to 2.0."
            )
        self.use_enhance = use_enhance

    @staticmethod
    def set_num_frames(num_frames: int):
        global NUM_FRAMES
        NUM_FRAMES = num_frames

    @staticmethod
    def set_enhance_weight(enhance_weight: float):
        global ENHANCE_WEIGHT
        ENHANCE_WEIGHT = enhance_weight

    def _get_enhance_scores(self, query, key):
        img_q, img_k = query, key

        num_frames = get_num_frames()
        _, num_heads, ST, head_dim = img_q.shape
        spatial_dim = ST / num_frames
        spatial_dim = int(spatial_dim)

        query_image = rearrange(
            img_q,
            "B N (T S) C -> (B S) N T C",
            T=num_frames,
            S=spatial_dim,
            N=num_heads,
            C=head_dim,
        )
        key_image = rearrange(
            img_k,
            "B N (T S) C -> (B S) N T C",
            T=num_frames,
            S=spatial_dim,
            N=num_heads,
            C=head_dim,
        )

        return enhance_score(query_image, key_image, head_dim, num_frames)

    def process_with_kv_cache(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        
    
        if attn.kv_cache is None:
            hidden_states_main, hidden_states_ip = (
                hidden_states[:, : -attn.cond_size],
                hidden_states[:, -attn.cond_size :],
            )
            
            
            rotary_emb_split_point = rotary_emb.shape[2] - attn.cond_size
            rotary_emb_main = rotary_emb[:, :, :rotary_emb_split_point]
            rotary_emb_ip = rotary_emb[:, :, rotary_emb_split_point:]
            

            
            query_main = attn.to_q(hidden_states_main)
            key_main = attn.to_k(hidden_states_main)
            value_main = attn.to_v(hidden_states_main)
            
            if attn.norm_q is not None:
                query_main = attn.norm_q(query_main)
            if attn.norm_k is not None:
                key_main = attn.norm_k(key_main)
                
            rotary_emb_main = rotary_emb_main.squeeze(0).transpose(0, 1)
            rotary_emb_ip = rotary_emb_ip.squeeze(0).transpose(0, 1)


            query_main = rope_apply_ip_adapter(query_main, rotary_emb_main, attn.heads)
            key_main = rope_apply_ip_adapter(key_main, rotary_emb_main, attn.heads)
            
 
            query_ip = attn.to_q(hidden_states_ip) + attn.add_q_lora(hidden_states_ip)
            key_ip = attn.to_k(hidden_states_ip) + attn.add_k_lora(hidden_states_ip)
            value_ip = attn.to_v(hidden_states_ip) + attn.add_v_lora(hidden_states_ip)
            
            
            if attn.norm_q is not None:
                query_ip = attn.norm_q(query_ip) 
            if attn.norm_k is not None:
                key_ip = attn.norm_k(key_ip)
                
            query_ip = rope_apply_ip_adapter(query_ip, rotary_emb_ip, attn.heads)
            key_ip = rope_apply_ip_adapter(key_ip, rotary_emb_ip, attn.heads)
            
            attn.kv_cache = {
                "key_ip": key_ip.detach(),
                "value_ip": value_ip.detach(),
            }

            full_key = torch.concat([key_main, key_ip], dim=1)
            full_value = torch.concat([value_main, value_ip], dim=1)

            
            query_main = query_main.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            full_key = full_key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            full_value = full_value.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            
            query_ip = query_ip.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            key_ip = key_ip.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            value_ip = value_ip.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            
            cond_hidden_states = attention_register.call(
                query_ip,
                key_ip,
                value_ip,
                attn_mask=attention_mask,
                dropout_p=0.0,
                is_causal=False,
            ).transpose(1, 2)
            
            hidden_states_main = attention_register.call(
                query_main,
                full_key,
                full_value,
                attn_mask=attention_mask,
                dropout_p=0.0,
                is_causal=False,
            ).transpose(1, 2)

            hidden_states_main = hidden_states_main.flatten(2, 3)
            hidden_states_main = hidden_states_main.type_as(query_main)
            
            cond_hidden_states = cond_hidden_states.flatten(2, 3)
            cond_hidden_states = cond_hidden_states.type_as(query_main)
            
            hidden_states = torch.concat(
                [hidden_states_main, cond_hidden_states], dim=1
            )
   
            hidden_states = attn.to_out[0](hidden_states)
            hidden_states = attn.to_out[1](hidden_states)
            
            return hidden_states

        else:
            key_ip = attn.kv_cache["key_ip"]
            value_ip = attn.kv_cache["value_ip"]

            query_main = attn.to_q(hidden_states)
            key_main = attn.to_k(hidden_states)
            value_main = attn.to_v(hidden_states)
            
            if attn.norm_q is not None:
                query_main = attn.norm_q(query_main)
            if attn.norm_k is not None:
                key_main = attn.norm_k(key_main)
                
            
            rotary_emb = rotary_emb.squeeze(0).transpose(0, 1)
            query_main = rope_apply_ip_adapter(query_main, rotary_emb, attn.heads)
            key_main = rope_apply_ip_adapter(key_main, rotary_emb, attn.heads)

            full_key = torch.concat([key_main, key_ip], dim=1)
            full_value = torch.concat([value_main, value_ip], dim=1)
            
            query_main = query_main.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            full_key = full_key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            full_value = full_value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

            hidden_states = attention_register.call(
                query_main,
                full_key,
                full_value,
                attn_mask=attention_mask,
                dropout_p=0.0,
                is_causal=False,
            ).transpose(1, 2)
            
            hidden_states = hidden_states.flatten(2, 3)
            hidden_states = hidden_states.type_as(query_main)

            hidden_states = attn.to_out[0](hidden_states)
            hidden_states = attn.to_out[1](hidden_states)

            return hidden_states

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[torch.Tensor] = None,
        no_cache: bool = False,
    ) -> torch.Tensor:
        
        if attn.cond_size is not None and not no_cache:
            return self.process_with_kv_cache(
                attn, hidden_states, attention_mask, rotary_emb
            )

        encoder_hidden_states_img = None
        if attn.add_k_proj is not None:
            # 512 is the context length of the text encoder, hardcoded for now
            image_context_length = encoder_hidden_states.shape[1] - 512
            encoder_hidden_states_img = encoder_hidden_states[:, :image_context_length]
            encoder_hidden_states = encoder_hidden_states[:, image_context_length:]
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        if (
            attn.to_q.weight.dtype != hidden_states.dtype
            and attn.to_q.weight.dtype != torch.int8
            and attn.to_q.weight.dtype != torch.uint8
        ):
            hidden_states = hidden_states.to(attn.to_q.weight.dtype)
        if (
            attn.to_k.weight.dtype != encoder_hidden_states.dtype
            and attn.to_k.weight.dtype != torch.int8
            and attn.to_k.weight.dtype != torch.uint8
        ):
            encoder_hidden_states = encoder_hidden_states.to(attn.to_k.weight.dtype)

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

        if rotary_emb is not None:
            query = apply_rotary_emb(query, rotary_emb)
            key = apply_rotary_emb(key, rotary_emb)

        if self.use_enhance:
            enhance_scores = self._get_enhance_scores(query, key)

        # I2V task
        hidden_states_img = None
        if encoder_hidden_states_img is not None:
            key_img = attn.add_k_proj(encoder_hidden_states_img)
            key_img = attn.norm_added_k(key_img)
            value_img = attn.add_v_proj(encoder_hidden_states_img)

            key_img = key_img.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            value_img = value_img.unflatten(2, (attn.heads, -1)).transpose(1, 2)

            hidden_states_img = attention_register.call(
                query,
                key_img,
                value_img,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
            ).transpose(1, 2)

            hidden_states_img = hidden_states_img.flatten(2, 3)
            hidden_states_img = hidden_states_img.type_as(query)

        hidden_states = attention_register.call(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        ).transpose(1, 2)

        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.type_as(query)

        if hidden_states_img is not None:
            hidden_states = hidden_states + hidden_states_img

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if self.use_enhance:
            hidden_states = hidden_states * enhance_scores

        return hidden_states
