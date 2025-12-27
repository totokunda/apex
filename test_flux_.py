from dotenv import load_dotenv
load_dotenv()
from src.quantize.load import load_gguf
import torch
from src.converters.transformer_converters import Flux2TransformerConverter
from accelerate import init_empty_weights
from src.transformer.flux2.base.model import Flux2Transformer2DModel
from src.quantize.ggml_layer import patch_model_from_state_dict as patch_model_ggml_from_state_dict
file_path = "/home/tosin_coverquick_co/apex-diffusion/components/8bd05a70ac56254734fd7403cb97cea823405b2d2f4d32fe3d1dcbd667b4ea24_flux2-dev-Q3_K_M.gguf"
state_dict, _ = load_gguf(file_path, type="transformer", dequant_dtype=torch.bfloat16, device="cpu")

converter = Flux2TransformerConverter()
converter.convert(state_dict)

with init_empty_weights():
    config = {
  "_class_name": "Flux2Transformer2DModel",
  "_diffusers_version": "0.36.0.dev0",
  "attention_head_dim": 128,
  "axes_dims_rope": [
    32,
    32,
    32,
    32
  ],
  "eps": 1e-06,
  "in_channels": 128,
  "joint_attention_dim": 15360,
  "mlp_ratio": 3.0,
  "num_attention_heads": 48,
  "num_layers": 8,
  "num_single_layers": 48,
  "out_channels": None,
  "patch_size": 1,
  "rope_theta": 2000,
  "timestep_guidance_channels": 256
}

    model = Flux2Transformer2DModel.from_config(config)

patch_model_ggml_from_state_dict(model, state_dict)
model.load_state_dict(state_dict, assign=True)
# model.language_model.layers.0.self_attn.q_proj.weight
# model.language_model.model.layers.0.self_attn.q_proj.weightdouble_blocks.0.txt_mlp.0