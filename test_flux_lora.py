from dotenv import load_dotenv
load_dotenv()
from src.quantize.load import load_gguf
from src.utils.safetensors import load_safetensors
import torch
from transformers import Mistral3ForConditionalGeneration, AutoConfig
from src.quantize.ggml_layer import patch_model_from_state_dict as patch_model_ggml_from_state_dict
from src.quantize.scaled_layer import patch_fpscaled_model_from_state_dict
from src.converters.text_encoder_converters import MistralTextEncoderConverter
from accelerate import init_empty_weights
file_path = "/home/tosin_coverquick_co/apex-diffusion/components/31089e6e194a10b16eae3097ecc5c035443bf09a34a5e9ab893f4d48e5a00d71_mistral_3_small_flux2_fp8.safetensors"
file_path_vision = "/home/tosin_coverquick_co/apex-diffusion/components/c502895883668a329ae0e00ea0d2771a6acf4c1f343cccff9479a47bcfe34baa_mmproj-BF16.gguf"
state_dict_language_model = load_safetensors(file_path, device="cpu")
state_dict_vision, _ = load_gguf(file_path_vision, type="text_encoder", key_map="mistral", dequant_dtype=torch.bfloat16, device="cpu")

converter = MistralTextEncoderConverter()

with init_empty_weights():
    config = AutoConfig.from_pretrained("black-forest-labs/FLUX.2-dev", subfolder="text_encoder")
    config.text_config.num_hidden_layers = 30
    model = Mistral3ForConditionalGeneration(config)

converter.convert(state_dict_language_model)
patch_fpscaled_model_from_state_dict(model, state_dict_language_model)
model.load_state_dict(state_dict_language_model, assign=True, strict=False)

converter.convert(state_dict_vision)
patch_model_ggml_from_state_dict(model, state_dict_vision)
model.load_state_dict(state_dict_vision, assign=True, strict=False)

for key, value in model.state_dict().items():
    if value.device == torch.device("meta"):
        print(key, value.shape)