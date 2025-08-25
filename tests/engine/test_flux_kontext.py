import os
os.environ["APEX_HOME_DIR"] = "/mnt/localssd"
from src.engine import create_engine
import torch

engine = create_engine("flux", "flux-dev-kontext-1-0-0-v1", "kontext", attention_type="sdpa")

prompt = "Give the guy a black fedora"
image = "/home/tosinkuye/apex/assets/image/couple.jpg"
video = engine.run(
    image=image,
    prompt=prompt,
    num_images=1,
    num_inference_steps=50,
    guidance_scale=2.5,
    resize_to_preferred_resolution=False,
    generator=torch.Generator(device="cuda").manual_seed(42)
)
video[0][0].save("test_flux_kontext_1_0_0_couple.png")   
#export_to_video(video[0], "test_flux_kontext_1_0_0_dog.mp4", fps=16, quality=8)