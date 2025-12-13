from diffusers import WanPipeline, UniPCMultistepScheduler
import torch
from diffusers.utils import export_to_video

device = "cuda"
pipe = WanPipeline.from_pretrained("yetter-ai/Wan2.2-TI2V-5B-Turbo-Diffusers", torch_dtype=torch.bfloat16).to(device)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=5.0)

width = 1280
height = 704
num_frames = 121
prompt = "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."

with torch.inference_mode():
    video = pipe(
        prompt = prompt,
        guidance_scale = 1.0,
        num_inference_steps = 4,
        generator = torch.Generator(device=device).manual_seed(42),
        width = width,
        height = height,
        num_frames = num_frames,
    ).frames[0]

    export_to_video(video, "video.mp4", fps=24)
