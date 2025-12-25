
from src.transformer.wan.base.model import WanTransformer3DModel
from src.converters.transformer_converters import FluxTransformerConverter
from src.lora.lora_converter import LoraConverter
from src.lora.manager import LoraManager
from safetensors.torch import load_file
from src.converters.convert import strip_common_prefix
from src.engine import UniversalEngine
yaml_path = "/home/tosin_coverquick_co/apex/manifest/verified/video/wan-2.2-a14b-text-to-video-1.0.0.v1.yml"
engine = UniversalEngine(yaml_path=yaml_path, components_to_load=["transformer"], auto_apply_loras=False, selected_components={
    "high_noise_transformer": {
        "variant": "FP8",
    },
    "low_noise_transformer": {
        "variant": "FP8",
    },
})
lora_path = "/home/tosin_coverquick_co/apex-diffusion/loras/439d28b80e5d2fdabbd68e72b1c9cd1f1f792e5f7cf7a68c34b1af16aca2db5f_high_noise_model.safetensors"

weights = load_file(lora_path)
manager = LoraManager()

manager.load_into(engine.transformer, [lora_path], adapter_names=["high_noise_lightning_lora"])
