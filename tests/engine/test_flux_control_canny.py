import os
os.environ["APEX_HOME_DIR"] = "/mnt/localssd"
from src.engine import create_engine
import torch
from src.preprocess import CannyPreprocessor
from diffusers.utils import load_image

engine = create_engine("flux", "flux-dev-control-canny-1-0-0-v1", "control", attention_type="sdpa")

prompt = "A robot made of exotic candies and chocolates of different kinds. The background is filled with confetti and celebratory gifts."
control_image = load_image("control_image.png")


video = engine.run(
    control_image=control_image,
    prompt=prompt,
    height=1024,
    width=1024,
    num_images=1,
    num_inference_steps=50,
    guidance_scale=30.0,
    generator=torch.Generator(device="cuda").manual_seed(42)
)
video[0][0].save("test_flux_control_canny_1_0_0_robot.png")   
#export_to_video(video[0], "test_flux_kontext_1_0_0_dog.mp4", fps=16, quality=8)