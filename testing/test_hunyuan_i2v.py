from src.engine import create_engine
from diffusers.utils import export_to_video
import torch

engine = create_engine("hunyuan", "/workspace/apex/manifest/hunyuan_i2v_13b.yml", "i2v", save_path="/dev/shm/models", attention_type="sdpa")

prompt = "A young woman reclines on crisp white sheets, her eyes flashing playful disbelief before her lips curl into a knowing smile—and she offers a single, flirtatious wink as her braids fall softly across her shoulder.."
image = "image.jpg"

video = engine.run(
    image=image,
    prompt=prompt,
    height=832,
    width=480,
    duration=61,
    num_videos=1,
    num_inference_steps=50,
    generator=torch.Generator(device="cuda").manual_seed(420)
)

export_to_video(video[0], "test_hunyuan_i2v_gina.mp4", fps=15)