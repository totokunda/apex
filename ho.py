
import torch
from glob import glob
import os
from safetensors.torch import load_file
from safetensors import safe_open
from tqdm import tqdm
from fastsafetensors import fastsafe_open
load_device = "cuda"
from src.transformer.wan.base.model import WanTransformer3DModel
from accelerate import init_empty_weights
path = "/home/tosin_coverquick_co/apex/wan_transformer_bf16"

files = glob(os.path.join(path, "*.safetensors"))
import time
start_time = time.perf_counter()
with init_empty_weights():
    config = WanTransformer3DModel.load_config(path)
    model = WanTransformer3DModel.from_config(config)

for file in tqdm(files):
    state_dict = {}
    #state_dict = load_file(file, device=load_device)
    with safe_open(file, framework="pt", device=load_device) as f:
        for key in f.keys():
            state_dict[key] = f.get_tensor(key)
        
    #model.load_state_dict(state_dict, assign=True, strict=False)

# stream = torch.cuda.Stream()
# with torch.cuda.stream(stream):
#     model.to(load_device, non_blocking=True)
# torch.cuda.synchronize()

print(time.perf_counter() - start_time, "seconds")