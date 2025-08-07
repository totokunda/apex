from src.engine import create_engine
from diffusers.utils import export_to_video
import torch

engine = create_engine("magi", "manifest/magi/magi_x2v_4_5b.yml", "v2v", save_path="/workspace/models", attention_type="flash", components_to_load=['vae'], component_dtypes={"text_encoder": torch.float32})

video = '/workspace/apex/MAGI-1/example/assets/prefix_video.mp4'
prompt = "Good Boy"

video = engine.run(
    prompt=prompt,
    video=video,
    height=480,
    width=832,
    duration=96,
    num_videos=1,
    num_inference_steps=32,
    seed=1234
)

export_to_video(video[0], "test_magi_v2v.mp4", fps=24, quality=8)