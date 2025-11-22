import torch
from diffusers import Kandinsky5T2VPipeline
from diffusers.utils import export_to_video

# Load the pipeline
model_id = "/home/tosin_coverquick_co/apex-diffusion/components/kandinskylab_Kandinsky-5.0-T2V-Pro-sft-5s-Diffusers"
pipe = Kandinsky5T2VPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
pipe = pipe.to("cuda")
pipe.transformer.set_attention_backend(
    "flex"
)                                       # <--- Sett attention bakend to Flex

# Generate video
#prompt = """On a wet street corner in a cyberpunk city at night, a large neon sign reading "I Love Gina" lights up sequentially, illuminating the dark, rainy environment with a pinkish-purple glow. The scene is a dark, rain-slicked street corner in a futuristic, cinematic cyberpunk city. Mounted on the metallic, weathered facade of a building is a massive, unlit neon sign. The sign's glass tube framework clearly spells out the words "I Love Gina". Initially, the street is dimly lit, with ambient light from distant skyscrapers creating shimmering reflections on the wet asphalt below. Then, the camera zooms in slowly toward the sign. As it moves, a low electrical sizzling sound begins. In the background, the dense urban landscape of the cyberpunk metropolis is visible through a light atmospheric haze, with towering structures adorned with their own flickering advertisements. A complex web of cables and pipes crisscrosses between the buildings. The shot is at a low angle, looking up at the sign to emphasize its grand scale. The lighting is high-contrast and dramatic, dominated by the neon glow which creates sharp, specular reflections and deep shadows. The atmosphere is moody and tech-noir. The overall video presents a cinematic photography realistic style."""

prompt = "A cat and a dog baking a cake together in a kitchen."
negative_prompt = "Static, 2D cartoon, cartoon, 2d animation, paintings, images, worst quality, low quality, ugly, deformed, walking backwards"

output = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    height=512,
    width=768,
    num_frames=121,  # ~5 seconds at 24fps
    num_inference_steps=50,
    guidance_scale=5.0,
).frames[0]

export_to_video(output, "output_kandinsky_d.mp4", fps=24, quality=9)