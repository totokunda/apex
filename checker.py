import torch
m = "/home/tosin_coverquick_co/apex-diffusion/components/e500a39d5b2035186e27b594c1e631a782c1ad10575c23176d495d3841a40d6e_step20000.ckpt"

state_dict = torch.load(m, map_location="cpu")
print(state_dict.keys())