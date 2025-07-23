from src.engine import create_engine
from diffusers.utils import export_to_video, load_image
import torch

engine = create_engine("cogvideo", "manifest/cogvideo/cogvideo_fun_control_5b.yml", "fun_control", save_path="/dev/shm/models", attention_type="sdpa")

# prompt = "A man with long black hair plays a red electric guitar."
control_video = "/workspace/VideoX-Fun/asset/pose.mp4"
prompt = "A young woman with beautiful face, dressed in white, is moving her body."
negative_prompt = "The video is not of a high quality, it has a low resolution. Watermark present in each frame. The background is solid. Strange body and strange trajectory. Distortion. "

video = engine.run(
    control_video=control_video,
    prompt=prompt,
    negative_prompt=negative_prompt,
    height=672,
    width=384,
    duration=49,
    num_videos=1,
    guidance_scale=6,
    num_inference_steps=50,
    generator=torch.Generator(device="cuda").manual_seed(43)
)

export_to_video(video[0], "fun_cogvideo_output_1.mp4", fps=8)