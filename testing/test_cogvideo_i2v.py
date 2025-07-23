from src.engine import create_engine
from diffusers.utils import export_to_video, load_image
import torch

engine = create_engine("cogvideo", "manifest/cogvideo/cogvideo_i2v_5b.yml", "i2v", save_path="/dev/shm/models", attention_type="sdpa")


prompt = "A man with short gray hair plays a red electric guitar."
image = load_image(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/guitar-man.png"
)


video = engine.run(
    image=image,
    prompt=prompt,
    height=480,
    width=832,
    duration=81,
    num_videos=1,
    guidance_scale=6,
    num_inference_steps=50,
    generator=torch.Generator(device="cuda").manual_seed(42)
)

export_to_video(video[0], "i2v_cogvideo_output_0.mp4", fps=8)