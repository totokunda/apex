import torch
import json
import os
import time
from glob import glob
from concurrent.futures import ThreadPoolExecutor
from safetensors.torch import load_file
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device
from transformers import UMT5EncoderModel

# Path and Config
path = "/home/tosin_coverquick_co/apex-diffusion/components/Wan-AI_Wan2.1-I2V-14B-480P-Diffusers/text_encoder"
target_dtype = torch.bfloat16
device = "cuda"
start_time = time.perf_counter()

model = UMT5EncoderModel.from_pretrained(path, torch_dtype=target_dtype, device_map="auto")
print(time.perf_counter() - start_time, "seconds")