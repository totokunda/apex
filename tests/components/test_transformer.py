from src.transformer.wan.base.model import WanTransformer3DModel
from src.quantize import patch_model, load_gguf
from accelerate import init_empty_weights

weights, _ = load_gguf("./wan_1.3b.Q6_K.gguf", type="transformer")

with init_empty_weights():
    config = WanTransformer3DModel.load_config("apex-diffusion/components/Wan-AI_Wan2.1-T2V-1.3B-Diffusers", subfolder="transformer")
    print(config)
    model = WanTransformer3DModel.from_config(config)


patch_model(model)
model.load_state_dict(weights, assign=True)

print(model)