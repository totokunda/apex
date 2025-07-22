from src.engine import create_engine
from diffusers.utils import export_to_video
import torch

engine = create_engine("cogvideo", "manifest/cogvideo/cogvideo_t2v_5b.yml", "t2v", save_path="/dev/shm/models", attention_type="sdpa")

prompt = "A panda, dressed in a small, red jacket and a tiny hat, sits on a wooden stool in a serene bamboo forest. The panda's fluffy paws strum a miniature acoustic guitar, producing soft, melodic tunes. Nearby, a few other pandas gather, watching curiously and some clapping in rhythm. Sunlight filters through the tall bamboo, casting a gentle glow on the scene. The panda's face is expressive, showing concentration and joy as it plays. The background includes a small, flowing stream and vibrant green foliage, enhancing the peaceful and magical atmosphere of this unique musical performance."

video = engine.run(
    prompt=prompt,
    height=480,
    width=832,
    duration=81,
    num_videos=1,
    guidance_scale=6.0,
    num_inference_steps=50,
    generator=torch.Generator(device="cuda").manual_seed(42)
)

export_to_video(video[0], "output_0.mp4", fps=8)