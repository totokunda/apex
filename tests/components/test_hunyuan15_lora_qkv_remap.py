import torch

from src.converters.transformer_converters import HunyuanVideo15TransformerConverter


def test_hunyuan15_double_blocks_qkv_lora_down_is_shared_and_lora_up_is_chunked():
    """
    For fused QKV LoRA:
      - lora_down (rank x in_dim) is shared across q/k/v
      - lora_up   (3*out_dim x rank) is split into q/k/v along the fused (out) dim
    """
    converter = HunyuanVideo15TransformerConverter()

    down = torch.randn(32, 2048, dtype=torch.bfloat16)
    up = torch.randn(6144, 32, dtype=torch.bfloat16)

    down_key = "diffusion_model.double_blocks.0.img_attn.qkv.lora_down.weight"
    up_key = "diffusion_model.double_blocks.0.img_attn.qkv.lora_up.weight"

    sd = {down_key: down, up_key: up}

    # Mimic the converter's in-place iteration strategy
    for k in list(sd.keys()):
        converter.remap_double_blocks_(k, sd)

    # Original fused keys are removed
    assert down_key not in sd
    assert up_key not in sd

    # Down is duplicated (shared) across q/k/v
    assert torch.equal(sd["diffusion_model.double_blocks.0.img_attn_q.lora_down.weight"], down)
    assert torch.equal(sd["diffusion_model.double_blocks.0.img_attn_k.lora_down.weight"], down)
    assert torch.equal(sd["diffusion_model.double_blocks.0.img_attn_v.lora_down.weight"], down)

    # Up is chunked into q/k/v along the fused output dim
    uq, uk, uv = torch.chunk(up, 3, dim=0)
    assert torch.equal(sd["diffusion_model.double_blocks.0.img_attn_q.lora_up.weight"], uq)
    assert torch.equal(sd["diffusion_model.double_blocks.0.img_attn_k.lora_up.weight"], uk)
    assert torch.equal(sd["diffusion_model.double_blocks.0.img_attn_v.lora_up.weight"], uv)


