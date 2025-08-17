import os
os.environ['HF_HOME'] = "/data/.hf_home"
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = "1"

import torch
from diffusers import AutoencoderKLWan, WanVACEPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from diffusers.utils import export_to_video, load_image

model_id = "Wan-AI/Wan2.1-VACE-14B-diffusers"
vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
pipe = WanVACEPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.bfloat16)
flow_shift = 3.0  # 5.0 for 720P, 3.0 for 480P
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=flow_shift)
pipe.to("cuda")

prompt = "Wide aerial over a Latium limestone range near ancient Rome (c. 60–70 CE): sunlit pale cliffs, Mediterranean pines and cypress, faint goat paths, long golden-hour shadows, soft haze; far on the horizon, a tiny city by the Tiber with low temples and aqueduct arches, unobtrusive. Camera performs a smooth dolly-in with optical zoom from ~24 mm to ~85 mm toward the jagged summit; subtle parallax over ridges; stabilized, no wobble; end angle slightly low, looking up. On the crown, the referenced dragon—black scales veined with molten cracks, horn layout and proportions exactly as ref—perches with wings partly unfurled. As the camera nears, the dragon inhales; chest and throat sacs brighten; embers leak between teeth. It bellows: a column of fire blasts forward with bright core and rolling orange bloom, sparks, heat distortion, and smoke shearing leeward, casting strobing highlights across limestone and armor-like scales. Wind whips wing membranes; flicker lighting plays across the face; shallow depth of field near the end. The dragon keeps its gaze fixed onward past the camera toward the distant horizon/city, dominant and unafraid."
negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"

images = [
    load_image("/data/apex/workflows/roman_city_dragon/assets/reference_anything/dragon_reference_anything.png"),
    load_image("/data/apex/workflows/roman_city_dragon/assets/reference_anything/mountain_reference_anything.png"),
]

height = 480
width = 832
num_frames = 81

output = pipe(
    reference_images=images,
    prompt=prompt,
    negative_prompt=negative_prompt,
    height=height,
    width=width,
    num_frames=num_frames,
    num_inference_steps=30,
    generator=torch.Generator().manual_seed(42),
    guidance_scale=5.0,
).frames[0]

export_to_video(output, "vace_anything_output.mp4", fps=16)
