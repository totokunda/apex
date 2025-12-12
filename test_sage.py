from sageattention import sageattn
import flash_attn 
import torch
from torch.nn.functional import scaled_dot_product_attention
B = 1
S = 1024
N = 40
D = 128

shape = (B, N, S, D)
q = torch.randn(*shape, dtype=torch.bfloat16).cuda()
k = torch.randn(*shape, dtype=torch.bfloat16).cuda()
v = torch.randn(*shape, dtype=torch.bfloat16).cuda()

sage_out = sageattn(q, k, v)
torch_out = scaled_dot_product_attention(q, k, v)
flash_attn_out = flash_attn.flash_attn_func(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2)

# print cosine similarity
print(torch.cosine_similarity(torch_out, sage_out, dim=0).mean())
print(torch.cosine_similarity(torch_out, flash_attn_out, dim=0).mean())