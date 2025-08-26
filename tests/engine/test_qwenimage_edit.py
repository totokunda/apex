import os
os.environ["APEX_HOME_DIR"] = "/mnt/localssd"
from src.engine import create_engine
import torch
from diffusers.utils import export_to_video

engine = create_engine("qwenimage", "qwenimage-edit-1-0-0-v1", "edit", attention_type="sdpa")

prompt = "Give the guy a black fedora"
negative_prompt=" "
image = "assets/image/couple.jpg"
video = engine.run(
    image=image,
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_images=1,
    num_inference_steps=50,
    true_cfg_scale=4.0,
    generator=torch.Generator(device="cuda").manual_seed(42)
)
video[0][0].save("test_qwenimage_edit_1_0_0_couple.png")   
#export_to_video(video[0], "test_qwenimage_edit_1_0_0_dog.mp4", fps=16, quality=8)