from src.converters.transformer_converters import HunyuanVideo15TransformerConverter, LoraTransformerConverter
from safetensors.torch import load_file

converter = HunyuanVideo15TransformerConverter()
lora_converter = LoraTransformerConverter()
state_dict = load_file("/home/tosin_coverquick_co/apex-diffusion/loras/6df3f6bdd81088998b6756d9ed5759ed6c87e8a94f69a348462f269e1b79f2d3_hunyuanvideo1.5_t2v_480p_lightx2v_4step_lora_rank_32_bf16.safetensors")

converter.convert(state_dict)
lora_converter.convert(state_dict)

out_dict = {} 
for key, value in state_dict.items():
    out_dict[key.replace("diffusion_model.", "")] = value

for key, value in out_dict.items():
    print(key, value.shape)