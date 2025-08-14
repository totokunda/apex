import torch
from src.mlx.transformer.wan.base.model import WanTransformer3DModel as WanTransformer3DModelMLX
from src.transformer.wan.base.model import WanTransformer3DModel as WanTransformer3DModelTorch
import torch 
from src.utils.mlx import torch_to_mlx, mlx_to_torch
from diffusers import UniPCMultistepScheduler
from src.mlx.scheduler.unipc import UniPCMultistepScheduler as UniPCMultistepSchedulerMLX

mlx_model = WanTransformer3DModelMLX.from_pretrained("/Users/tosinkuye/apex-diffusion/components/Wan-AI_Wan2.1-T2V-1.3B-Diffusers/transformer/mlx", dtype='float16')
torch_model = WanTransformer3DModelTorch.from_pretrained("/Users/tosinkuye/apex-diffusion/components/Wan-AI_Wan2.1-T2V-1.3B-Diffusers/transformer", torch_dtype=torch.float16).to("mps")

mlx_model = mlx_model.eval()
torch_model = torch_model.eval()
file = "denoise_input.pt"

tensors = torch.load(file)
t_hidden_states = tensors["hidden_states"].to("mps")
t_timestep = tensors["timestep"].to("mps")
t_t = t_timestep.squeeze()
t_encoder_hidden_states = tensors["encoder_hidden_states"].to("mps")

mlx_tensors = torch_to_mlx(tensors.copy())
m_hidden_states = mlx_tensors["hidden_states"]
m_timestep = mlx_tensors["timestep"]
m_t = m_timestep.squeeze()
m_encoder_hidden_states = mlx_tensors["encoder_hidden_states"]

scheduler = UniPCMultistepScheduler.from_pretrained('/Users/tosinkuye/apex-diffusion/configs/Wan-AI/Wan2.1-T2V-1.3B-Diffusers/resolve/main/scheduler')
scheduler_mlx = UniPCMultistepSchedulerMLX.from_pretrained('/Users/tosinkuye/apex-diffusion/configs/Wan-AI/Wan2.1-T2V-1.3B-Diffusers/resolve/main/scheduler')
scheduler.set_timesteps(30)
scheduler_mlx.set_timesteps(30)

def compare_tensors(t_tensor, m_tensor, label: str = ""):
    if t_tensor is None or m_tensor is None:
        print("One of the tensors is None", label)
        return
    mt_tensor = mlx_to_torch(m_tensor)
    print(label, t_tensor.shape, mt_tensor.shape, t_tensor.dtype, mt_tensor.dtype, t_tensor.device, mt_tensor.device)
    try:
        torch.testing.assert_close(t_tensor, mt_tensor, atol=1e-4, rtol=1e2)
        print("Passed", label)
    except Exception as e:
        print("Failed", label, e)

with torch.no_grad():
   
    t_noise_pred = torch_model(hidden_states=t_hidden_states, encoder_hidden_states=t_encoder_hidden_states, timestep=t_timestep, return_dict=False)[0]
    m_noise_pred = mlx_model(hidden_states=m_hidden_states, encoder_hidden_states=m_encoder_hidden_states, timestep=m_timestep, return_dict=False)[0]
    
    t_latents = scheduler.step(t_noise_pred, t_t, t_hidden_states, return_dict=False)[0]
    m_latents = scheduler_mlx.step(m_noise_pred, m_t, m_hidden_states, return_dict=False)[0]
    
    compare_tensors(t_noise_pred, m_noise_pred, "noise_pred")
    compare_tensors(t_latents, m_latents, "latents")
    
    
    