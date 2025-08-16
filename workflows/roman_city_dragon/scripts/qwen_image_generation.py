from diffusers import DiffusionPipeline
import torch

model_name = "Qwen/Qwen-Image"

# Load the pipeline
if torch.cuda.is_available():
    torch_dtype = torch.bfloat16
    device = "cuda"
else:
    torch_dtype = torch.float32
    device = "cpu"

pipe = DiffusionPipeline.from_pretrained(model_name, torch_dtype=torch_dtype)
pipe = pipe.to(device)

positive_magic = {
    "en": "Ultra HD, 4K, cinematic composition.", # for english prompt,
    "zh": "超清，4K，电影级构图" # for chinese prompt,
}

# Generate image
prompt = '''Majestic mountain in the Latium countryside near ancient Rome (circa 60–70 CE), the mountain fills most of the frame with rugged pale limestone cliffs, sunlit slopes, Mediterranean pines and cypress, wildflowers and scrub, faint goat paths, warm golden-hour light with long shadows, soft haze and atmospheric perspective, rolling green hills in the mid-ground, far on the horizon a small ancient Roman city by the Tiber—low temples and forums, red-tile roofs, distant aqueduct arches—tiny and unobtrusive, no monumental amphitheater yet, serene sky with thin clouds, ultra-detailed, natural color, cinematic landscape photograph, wide-angle 28–35 mm, tripod-stable, f/8 depth and crisp textures, composition: mountain occupies ~65% of frame, Rome a subtle glimmer in the distance'''

negative_prompt = " " # using an empty string if you do not have specific concept to remove

# Generate with different aspect ratios
aspect_ratios = {
    "1:1": (1328, 1328),
    "16:9": (1664, 928),
    "9:16": (928, 1664),
    "4:3": (1472, 1140),
    "3:4": (1140, 1472),
    "3:2": (1584, 1056),
    "2:3": (1056, 1584),
}

width, height = aspect_ratios["16:9"]

image = pipe(
    prompt=prompt + positive_magic["en"],
    negative_prompt=negative_prompt,
    width=width,
    height=height,
    num_inference_steps=50,
    true_cfg_scale=4.0
).images[0]

image.save("assets/mountain.png")
