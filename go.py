from src.transformer.wan.base.model import WanTransformer3DModel
import json
import os
import time
import torch

path = "/home/tosin_coverquick_co/apex-diffusion/components/Wan-AI_Wan2.2-T2V-A14B-Diffusers/transformer"

start_time = time.perf_counter()
model = WanTransformer3DModel.from_pretrained(path, torch_dtype=torch.bfloat16, device_map="cuda")
print(time.perf_counter() - start_time, "seconds")
