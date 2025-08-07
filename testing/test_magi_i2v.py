from src.engine import create_engine
from diffusers.utils import export_to_video
import torch


engine = create_engine("magi", "manifest/magi/magi_x2v_4_5b.yml", "i2v", save_path="/workspace/models", attention_type="flash", components_to_load=['text_encoder', 'transformer'], component_dtypes={"text_encoder": torch.float32})

image = '/workspace/apex/MAGI-1/example/assets/image.jpeg'
prompt = "Good Boy"

video = engine.run(
    prompt=prompt,
    image=image,
    height=480,
    width=832,
    duration=96,
    num_videos=1,
    num_inference_steps=32,
    seed=1234
)

export_to_video(video[0], "test_magi_i2v_2.mp4", fps=24, quality=8)