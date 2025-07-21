from src.engine import create_engine
from diffusers.utils import export_to_video
import torch

engine = create_engine("hunyuan", "/workspace/apex/manifest/hunyuan_t2v_13b.yml", "t2v", save_path="/dev/shm/models", attention_type="sdpa")
engine = create_engine("hunyuan", "/workspace/apex/manifest/hunyuan_t2v_13b.yml", "t2v", save_path="/dev/shm/models", attention_type="sdpa")

prompt = "A beautiful woman in a flowing red dress walks through a dimly lit city street, neon lights reflecting off wet pavement, her silhouette glowing as the camera slowly tracks her from behind."

video = engine.run(
    prompt=prompt,
    height=480,
    width=832,
    duration=61,
    num_videos=1,
    num_inference_steps=50,
    generator=torch.Generator(device="cuda").manual_seed(69)
)

export_to_video(video[0], "test_hunyuan_t2v_woman1.mp4", fps=15)