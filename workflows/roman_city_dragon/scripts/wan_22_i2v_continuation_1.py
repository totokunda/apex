import sys
import os
sys.path.append('/data/apex')
os.environ["APEX_HOME_DIR"] = "/data"
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = "1"

from src.engine import create_engine
from diffusers.utils import export_to_video
from PIL import Image
import torch

engine = create_engine("wan", "/data/apex/workflows/roman_city_dragon/manifest/i2v.yml", "i2v", attention_type="flash", component_dtypes={
    "text_encoder":  torch.float32,
})

image = Image.open('assets/images/last_frame_blurred.png')
prompt = "The dragon continues to breath fire as the camera pans across the center of its face towards the right blowing some of the fire directly into the camera lens. The camera then pulls away from the dragon's face and pulls away to show the dragon's full body on the left. The dragon stops breathing fire and preps its wings to take flight. 8K quality, cinematic, steady stabilization, natural motion blur."
negative_prompt="lowres, text, error, cropped, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, username, watermark, signature"

video = engine.run(
    image=image,
    prompt=prompt,
    negative_prompt=negative_prompt,
    height=480,
    width=832,
    duration=81,
    num_videos=1,
    num_inference_steps=50,
    guidance_scale=[4.0, 3.0]
)

export_to_video(video[0], "assets/flux_kontext/dragon_flux_kontext_continuation_1.mp4", fps=16, quality=9)