import mlx.core as mx
import torch
from src.mlx.transformer.wan.base.model import WanTransformer3DModel
from src.utils.mlx import to_mlx
import time 
model = WanTransformer3DModel.from_pretrained("apex-diffusion/components/Wan-AI_Wan2.1-T2V-1.3B-Diffusers/transformer/mlx", dtype='float16')
model = model.eval()
file = "denoise_input.pt"

data = torch.load(file)

data['hidden_states'] = data['hidden_states'].repeat(1, 1, 2, 1, 1)[:, :, :21, :, :]
data['timestep'] = data['timestep']
data['encoder_hidden_states'] = data['encoder_hidden_states']

data = {k: to_mlx(v) if isinstance(v, torch.Tensor) else v for k, v in data.items()} 
torch.mps.empty_cache()

hidden_states = data["hidden_states"]
timestep = data["timestep"]
encoder_hidden_states = data["encoder_hidden_states"]

batch_size, num_channels, num_frames, height, width = hidden_states.shape
p_t, p_h, p_w = model.config.patch_size
post_patch_num_frames = num_frames // p_t
post_patch_height = height // p_h
post_patch_width = width // p_w

print(batch_size, num_channels, num_frames, height, width)
print(p_t, p_h, p_w)
print(post_patch_num_frames, post_patch_height, post_patch_width)

print(f"Peak memory: {mx.get_peak_memory() / 1024**3:.2f} GB")

start_time = time.time()
@mx.compile
def inference_model(hidden_states, timestep, encoder_hidden_states):
    out = model(
        hidden_states=hidden_states,
        timestep=timestep,
        encoder_hidden_states=encoder_hidden_states,
        encoder_hidden_states_image=None,
        return_dict=False,
    )[0]
    return out

out = inference_model(hidden_states, timestep, encoder_hidden_states)
mx.eval(out)

out_2 = inference_model(hidden_states, timestep, encoder_hidden_states / 2)
mx.eval(out_2)

end_time = time.time()
print(f"Time taken: {end_time - start_time:.2f} seconds")
print(f"Active memory: {mx.get_active_memory() / 1024**3:.2f} GB")
print(f"Peak memory: {mx.get_peak_memory() / 1024**3:.2f} GB")









