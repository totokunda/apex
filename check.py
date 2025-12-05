from safetensors.torch import load_file
from src.converters.convert import get_transformer_converter

state_dict = load_file("/home/tosin_coverquick_co/apex-diffusion/loras/97b0a50deeb21540c458aaa42a8c331776f0d9b50b9566b0d848562bcd2dffe7_1776890.safetensors")
convert = get_transformer_converter("wan.base")
convert.convert(state_dict)
print(state_dict.keys())