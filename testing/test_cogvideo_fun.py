from src.engine import create_engine
from diffusers.utils import export_to_video, load_image
import torch

engine = create_engine("cogvideo", "manifest/cogvideo/cogvideo_fun_5b.yml", "fun", save_path="/dev/shm/models", attention_type="sdpa")

prompt = "A man with long black hair plays a red electric guitar."
control_video = "i2v_cogvideo_output_0.mp4"

video = engine.run(
    control_video=control_video,
    prompt=prompt,
    height=480,
    width=832,
    duration=81,
    num_videos=1,
    guidance_scale=6,
    num_inference_steps=50,
    generator=torch.Generator(device="cuda").manual_seed(42)
)

export_to_video(video[0], "fun_cogvideo_output_0.mp4", fps=8)