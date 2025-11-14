from src.quantize.load import load_gguf
from src.quantize.ggml_layer import patch_model
from accelerate import init_empty_weights
from src.transformer.flux.base.model import FluxTransformer2DModel
from src.converters.convert import get_transformer_converter
path  = "/Users/tosinkuye/apex-workspace/apex/382ed0e097a9ed0a9d7c7b6318c3ede4ecde9d540dc1d187e66ef85864e63927_flux1-kontext-dev-Q2_K.gguf"
state_dict, qtype_dict = load_gguf(path, type="transformer")

with init_empty_weights():
    model = FluxTransformer2DModel.from_config({
      "_class_name": "FluxTransformer2DModel",
      "_diffusers_version": "0.34.0.dev0",
      "_name_or_path": "../checkpoints/flux-dev/transformer",
      "attention_head_dim": 128,
      "axes_dims_rope": [
        16,
        56,
        56
      ],
      "guidance_embeds": True,
      "in_channels": 64,
      "joint_attention_dim": 4096,
      "num_attention_heads": 24,
      "num_layers": 19,
      "num_single_layers": 38,
      "out_channels": None,
      "patch_size": 1,
      "pooled_projection_dim": 768
    }
    )
    
    converter = get_transformer_converter("flux.base")
    converter.convert(state_dict)
    model_state_dict = model.state_dict()
    patch_model(model)

model.load_state_dict(model_state_dict, assign=True)
print(model)