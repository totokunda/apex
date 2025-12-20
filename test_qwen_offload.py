from accelerate import cpu_offload
from diffusers.hooks import apply_group_offloading
import torch
import time
from tqdm import tqdm
from src.transformer.qwenimage.base.model import QwenImageTransformer2DModel
path = "/home/tosin_coverquick_co/apex-diffusion/components/Qwen_Qwen-Image/transformer"
    
model = QwenImageTransformer2DModel.from_pretrained(path, torch_dtype=torch.bfloat16)
apply_group_offloading(model, onload_device="cuda", offload_device="cpu", offload_type="block_level", num_blocks_per_group=5, non_blocking=True, use_stream=True, record_stream=True)

block_data = torch.load("block.safetensors")

hidden_states = block_data.pop("hidden_states")
encoder_hidden_states = block_data.pop("encoder_hidden_states")
with torch.no_grad():
    for block in tqdm(model.transformer_blocks):
        encoder_hidden_states, hidden_states = block(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            **block_data
        )