from dotenv import load_dotenv
import torch
load_dotenv()
from src.engine.registry import UniversalEngine
from diffusers.utils import export_to_video
yaml_path = "/home/tosin_coverquick_co/apex/manifest/engine/wan/ditto-vace-14b-1.0.0.v1.yml"
engine = UniversalEngine(yaml_path=yaml_path)

out = engine.run(
    video='/home/tosin_coverquick_co/apex/videos/whipping_pose.mp4',
    prompt="Add some fire and flame to the background",
    negative_prompt="lowres, text, error, cropped, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, username, watermark, signature",
    num_inference_steps=4,
    height=832,
    width=480,
    guidance_scale=5.0,
    num_frames=81
)

export_to_video(out[0], "output_ditto.mp4", fps=16)