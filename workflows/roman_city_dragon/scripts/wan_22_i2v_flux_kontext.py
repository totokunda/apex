import sys
sys.path.append('/workspace/apex')

import torch
from src.engine import create_engine
from diffusers.utils import export_to_video
from PIL import Image

engine = create_engine("wan", "/workspace/apex/manifest/wan/wan_22_i2v_14b.yml", "i2v", attention_type="flash", denoise_type="moe")

image = Image.open('assets/flux_kontext/dragon_flux_kontext.png')
prompt = "Wide aerial over a Latium limestone range near ancient Rome (c. 60–70 CE): sunlit pale cliffs, Mediterranean pines and cypress, faint goat paths, long golden-hour shadows, soft haze; far on the horizon, a tiny city by the Tiber with low temples and aqueduct arches, unobtrusive. Camera performs a smooth dolly-in with optical zoom from ~24 mm to ~85 mm toward the jagged summit; subtle parallax over ridges; stabilized, no wobble; end angle slightly low, looking up. On the crown, the referenced dragon—black scales veined with molten cracks, horn layout and proportions exactly as ref—perches with wings partly unfurled. As the camera nears, the dragon inhales; chest and throat sacs brighten; embers leak between teeth. It bellows: a column of fire blasts forward with bright core and rolling orange bloom, sparks, heat distortion, and smoke shearing leeward, casting strobing highlights across limestone and armor-like scales. Wind whips wing membranes; flicker lighting plays across the face; shallow depth of field near the end. The dragon keeps its gaze fixed onward past the camera toward the distant horizon/city, dominant and unafraid."
negative_prompt="lowres, text, error, cropped, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, username, watermark, signature"

video = engine.run(
    image=image,
    prompt=prompt,
    negative_prompt=negative_prompt,
    height=480,
    width=832,
    duration='5s',
    num_videos=1,
    num_inference_steps=40,
    guidance_scale=[4.0, 3.0]
)

export_to_video(video[0], "assets/flux_kontext/dragon_flux_kontext.mp4", fps=16, quality=9)