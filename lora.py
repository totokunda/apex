from safetensors.torch import load_file
from src.lora.lora_converter import LoraConverter
from src.converters.transformer_converters import FluxTransformerConverter

state_dict = load_file("super-realism.safetensors")
converter = LoraConverter()
converter.convert(state_dict)
flux_converter = FluxTransformerConverter()
flux_converter.convert(state_dict)
for key, value in state_dict.items():
    print(key, value.shape)