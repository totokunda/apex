import sys
import os
sys.path.append('/home/tosinkuye/apex')
path = "/home/tosinkuye/apex"

import cv2
import torch
torch.set_float32_matmul_precision('high')
from src.engine import create_engine
from diffusers.utils import export_to_video
from PIL import Image

engine = create_engine("wan", f"{path}/workflows/roman_city_dragon/manifest/vace.yml", "vace", attention_type="sage")
    
images = [
    Image.open(f'{path}/workflows/roman_city_dragon/assets/images/frame_49.png')
]

control_video = f"{path}/workflows/roman_city_dragon/assets/depth/depth_dragon_flux_kontext_continuation_1_cut.mp4"
# get the number of frames in the control video
control_video_path = f"{path}/workflows/roman_city_dragon/assets/depth/depth_dragon_flux_kontext_continuation_1_cut.mp4"
_video = cv2.VideoCapture(control_video_path)
num_frames = int(_video.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Number of frames in control video: {num_frames}")

prompt = "Wide aerial over a Latium limestone range near ancient Rome (c. 60–70 CE): sunlit pale cliffs, Mediterranean pines and cypress, faint goat paths, long golden-hour shadows, soft haze; far on the horizon, a tiny city by the Tiber with low temples and aqueduct arches, unobtrusive. Camera performs a smooth dolly-in with optical zoom from ~24 mm to ~85 mm toward the jagged summit; subtle parallax over ridges; stabilized, no wobble; end angle slightly low, looking up. On the crown, the referenced dragon—black scales veined with molten cracks, horn layout and proportions exactly as ref—perches with wings partly unfurled. As the camera nears, the dragon inhales; chest and throat sacs brighten; embers leak between teeth. It bellows: a column of fire blasts forward with bright core and rolling orange bloom, sparks, heat distortion, and smoke shearing leeward, casting strobing highlights across limestone and armor-like scales. Wind whips wing membranes; flicker lighting plays across the face; shallow depth of field near the end. The dragon keeps its gaze fixed onward past the camera toward the distant horizon/city, dominant and unafraid."
negative_prompt="lowres, text, error, cropped, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, username, watermark, signature"

video = engine.run(
    video=control_video,
    reference_images=images,
    prompt=prompt,
    height=480,
    width=832,
    duration=num_frames,
    num_videos=1,
    num_inference_steps=4,
    guidance_scale=5.0
)

export_to_video(video[0], f"{path}/workflows/roman_city_dragon/assets/control/vace_control_2.mp4", fps=16, quality=9)