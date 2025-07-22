from src.engine import create_engine
from diffusers.utils import export_to_video, load_image
import torch

engine = create_engine("hunyuan", "/workspace/apex/manifest/hunyuan_i2v_13b.yml", "i2v", save_path="/dev/shm/models", attention_type="sdpa")


prompt = "A man with short gray hair plays a red electric guitar."
image = load_image(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/guitar-man.png"
)

video = engine.run(
    image=image,
    prompt=prompt,
    height=480,
    width=832,
    duration=61,
    num_videos=1,
    num_inference_steps=50,
    generator=torch.Generator(device="cuda").manual_seed(420)
)

export_to_video(video[0], "test_hunyuan_i2v_gina.mp4", fps=15)