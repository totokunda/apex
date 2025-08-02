import torch 

module = torch.load('/workspace/models/components/tencent/HunyuanVideo-Avatar/resolve/main/ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt', map_location='cpu', mmap=True)

print(module['module'].keys())