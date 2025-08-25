import os
os.environ["APEX_HOME_DIR"] = "/mnt/localssd"
from src.engine import create_engine
import torch
from diffusers.utils import load_image

engine = create_engine("flux", "flux-dev-fill-1-0-0-v1", "fill", attention_type="sdpa")
image = load_image("https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/cup.png")
mask = load_image("https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/cup_mask.png")
prompt = "a white paper cup"
video = engine.run(
    image=image,
    mask_image=mask,
    prompt=prompt,
    height=1632,
    width=1232,
    num_images=1,
    num_inference_steps=50,
    guidance_scale=30,
    generator=torch.Generator(device="cuda").manual_seed(42)
)
video[0][0].save("test_flux_fill_1_0_0_cup.png")   
#export_to_video(video[0], "test_flux_kontext_1_0_0_dog.mp4", fps=16, quality=8)