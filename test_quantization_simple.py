from src.engine import create_engine
from diffusers.utils import export_to_video
import torch
from src.quantize.quantizer import ModelQuantizer

engine = create_engine("magi", "manifest/magi/magi_x2v_4_5b.yml", "t2v", save_path="/workspace/models", attention_type="flash", components_to_load=['transformer', 'vae', 'text_encoder'], component_dtypes={"text_encoder": torch.float32})

quantizer = ModelQuantizer(
    quant_method="gguf_config_q3_k",
    target_memory_gb=2.0,
    auto_optimize=True
)

transformer = engine.transformer

quantized_model = quantizer.quantize(transformer)

print(quantized_model)