from src.transformer.wan.base.model import WanTransformer3DModel 
from src.converters.transformer_converters import WanTransformerConverter
from safetensors.torch import load_file
from accelerate import init_empty_weights
from src.quantize.scaled_layer import patch_fpscaled_model

with init_empty_weights():
    model = WanTransformer3DModel.from_config({
        "_class_name": "WanTransformer3DModel",
        "_diffusers_version": "0.35.0.dev0",
        "added_kv_proj_dim": None,
        "attention_head_dim": 128,
        "cross_attn_norm": True,
        "eps": 1e-06,
        "ffn_dim": 13824,
        "freq_dim": 256,
        "image_dim": None,
        "in_channels": 16,
        "num_attention_heads": 40,
        "num_layers": 40,
        "out_channels": 16,
        "patch_size": [
          1,
          2,
          2
        ],
        "pos_embed_seq_len": None,
        "qk_norm": "rms_norm_across_heads",
        "rope_max_seq_len": 1024,
        "text_dim": 4096
        }
    )
    patch_fpscaled_model(model)
    path = "/home/tosin_coverquick_co/apex-diffusion/components/d22bd8433943ec62b41672dc9a9e9d9a901197b912a4e4159b04b25908f79ea5_Wan2_2-T2V-A14B_HIGH_fp8_e4m3fn_scaled_KJ.safetensors"


state_dict = load_file(path)
WanTransformerConverter().convert(state_dict)
print(model.load_state_dict(state_dict, assign=True, strict=False))


