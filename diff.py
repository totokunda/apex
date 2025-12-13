import torch
import numpy as np
from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
from diffusers.utils import export_to_video, load_image

dtype = torch.bfloat16
device = "cuda"

model_id = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
pipe = WanImageToVideoPipeline.from_pretrained(model_id, vae=vae, torch_dtype=dtype)
pipe.to(device)

height = 704
width = 1280
num_frames = 121
num_inference_steps = 50
guidance_scale = 5.0

image = load_image("/home/tosin_coverquick_co/apex/images/wide.png")
prompt = "A young dancer performs energetic hip-hop moves in the middle of a busy street in Shibuya, Tokyo. Neon signs glow overhead, crowds flow around the dancer, blurred motion from passing pedestrians and traffic. Vibrant night lights reflect off wet pavement. Dynamic camera movement: smooth tracking shots, slight handheld realism. High-contrast cinematic lighting, rich colors, detailed urban textures. Atmosphere feels lively, modern, and electric, capturing the iconic Shibuya nightlife energy."
negative_prompt = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"

output = pipe(
    image=image,
    prompt=prompt,
    negative_prompt=negative_prompt,
    height=height,
    width=width,
    num_frames=num_frames,
    guidance_scale=guidance_scale,
    num_inference_steps=num_inference_steps,
    generator=torch.Generator(device=device).manual_seed(42),
).frames[0]
export_to_video(output, "5bit2v_output.mp4", fps=24)
