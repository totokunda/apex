from dotenv import load_dotenv
import torch
load_dotenv()
from src.engine.registry import UniversalEngine

yaml_path = "/home/tosin_coverquick_co/apex/manifest/engine/wan/magic-wan-2.2-14b-text-to-image-1.0.0.v1.yml"
engine = UniversalEngine(yaml_path=yaml_path)

prompt = """A cinematic space-themed scene set against a deep starfield and distant galaxies. A small group of people in sleek, futuristic attire stand together on a glowing platform floating in space, gazing outward with a sense of creativity and ambition. Above and slightly behind them, the text “Apex Studio” appears large and clearly readable, styled in modern, bold typography with a soft luminous glow. Dramatic lighting, subtle nebula colors, depth and scale, high detail but clean composition, professional sci-fi creative aesthetic."""

out = engine.run(
    prompt=prompt,
    height=1024,
    width=1536,
    seed=42,
    num_inference_steps=50,
)

out[0].save("output_magic_wan_image.png")