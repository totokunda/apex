from safetensors.torch import safe_open
import torch
path = "/home/tosin_coverquick_co/apex-diffusion/components/best_netG.pt"

contents = torch.load(path, map_location="meta", weights_only=True, mmap=True)

gen = contents["generator"]
for key, value in gen.items():
    print(key, value.shape, value.dtype)