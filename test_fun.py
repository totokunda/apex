from dotenv import load_dotenv
import torch
load_dotenv()
from src.engine.registry import UniversalEngine
from diffusers.utils import export_to_video
yaml_path = "/home/tosin_coverquick_co/apex/manifest/engine/wan/wan-fun-control-2.2-1.0.0.v1.yml"
engine = UniversalEngine(yaml_path=yaml_path)

inps = {
    "reference_image": "/home/tosin_coverquick_co/apex/images/apex_woman_image.png",
    "control_video": "/home/tosin_coverquick_co/apex/man_jacket_snow_edit.mp4",
     "prompt": "A man with blonde hair and a beard hugging a tree",
    "negative_prompt": "lowres, text, error, cropped, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, username, watermark, signature",
     "resolution": 480,
     "aspect_ratio": 61,
     "num_inference_steps": 30,
     "height": 512,
     "width": 512,
     "seed": 2856437645,
     "high_noise_guidance_scale": 4,
     "low_noise_guidance_scale": 3,
     "boundary_ratio": 0.875,
     "duration": "5s"
}

out = engine.run(**inps)