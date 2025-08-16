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
prompt = '''Majestic mountain in the Latium countryside near ancient Rome (circa 60–70 CE), the mountain fills most of the frame with rugged pale limestone cliffs and jagged outcrops, wind-scoured terraces, Mediterranean pines and cypress bent and battered by a cold wind, scrub and wildflowers flattened into the slopes, faint goat paths cutting like scars. Golden-hour light is waning — long, cold shadows stretch across the rocks while a bruised, low sky thickens with heavy, thin thunderheads and a creeping mist pools in the valleys. A lone vulture wheels high above; a thin column of smoke rises on the distant horizon. Rolling green hills in the mid-ground, and far on the horizon a small ancient Roman town by the Tiber — low temples and forums, red-tile roofs and distant aqueduct arches — tiny and ominously unobtrusive (no monumental amphitheatre yet). Ultra-detailed, high-contrast cinematic landscape photograph, slightly desaturated natural palette, subtle film grain, dramatic composition, foreboding atmosphere.'''

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

images = pipe(
    num_images_per_prompt=4,
    prompt=prompt + positive_magic["en"],
    negative_prompt=negative_prompt,
    width=width,
    height=height,
    num_inference_steps=50,
    true_cfg_scale=4.0
).images

for i, image in enumerate(images):
    image.save(f"assets/ominous_mountain_{i}.png")
