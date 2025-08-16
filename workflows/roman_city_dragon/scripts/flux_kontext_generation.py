import torch
from diffusers import FluxKontextPipeline
from diffusers.utils import load_image

pipe = FluxKontextPipeline.from_pretrained("black-forest-labs/FLUX.1-Kontext-dev", torch_dtype=torch.bfloat16)
pipe.to("cuda")

input_image = load_image("assets/dragon.png")
width, height = input_image.size

images = pipe(
  num_images_per_prompt=4,
  image=input_image,
  height=height,
  width=width,
  prompt="The dragon perched on a jagged mountain summit, wings flared, neck arched, jaws open in a ferocious roar, embers glowing—menacing stance. Stormy sky, swirling mist, dramatic backlight/rim light, valley far below for scale. Dragon small in frame (farther away), top-third composition. Cinematic, 8K, ultra-detailed.",
  guidance_scale=2.5,
  num_inference_steps=50,
).images

for i, image in enumerate(images):  
    image.save(f"assets/dragon_flux_kontext_{i}.png", quality=95, optimize=True)

