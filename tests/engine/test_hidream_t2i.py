import os
os.environ["APEX_HOME_DIR"] = "/mnt/localssd"
from src.engine import create_engine
import torch
from diffusers.utils import export_to_video

engine = create_engine("hidream", "hidream-i1-full-text-to-image-1-0-0-v1", "t2i")

prompt = "A swirling nebula with a rocket ship heading towards it in space. Realistic, 4K, cinematic composition."
negative_prompt="lowres, text, error, cropped, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, username, watermark, signature"

video = engine.run(
    prompt=prompt,
    negative_prompt=negative_prompt,
    height=1024,
    width=1024,
    num_images=1,
    num_inference_steps=50,
    guidance_scale=5.0,
    generator=torch.Generator(device="cuda").manual_seed(42)
)
video[0][0].save("test_hidream_t2i_1_0_0_nebula.png")
# export_to_video(video[0], "test_hidream_t2i_1_0_0_nebula.mp4", fps=16, quality=10)