from dotenv import load_dotenv
load_dotenv()
from diffusers.utils import export_to_video
from src.engine.registry import UniversalEngine

yaml_path = "/home/tosin_coverquick_co/apex/manifest/engine/ovis/ovis-text-to-image-1.0.0.v1.yml"
prompt = """
An autumn aesthetic vision-board collage arranged in a clean grid layout. Moody, cinematic tones with deep blues, warm browns, burnt orange, forest green, and soft gold. Subtle fall leaves scattered across the image.

Quadrants include:
– Minimal poster with text: “FEEL THE FEAR & DO IT ANYWAY!”
– A person at a desk surrounded by glowing screens in a circular room
– Tree bark texture with a centered quote: “God wouldn’t place a dream so big in your heart if you weren’t capable of making it real.”
– Cozy high-rise apartment at night overlooking city lights
– Elegant serif text across the center reading “October”
– Vintage sheet music on aged paper
– Dark ocean waves at night
– Nighttime city skyscrapers with glowing windows
– Futuristic city island with sci-fi interface elements
– Drummer practicing intensely with text: “OBSESSION beats talent.”

Cinematic, nostalgic yet modern, motivational, editorial collage, soft grain, cohesive aesthetic, high-resolution wallpaper.
"""

engine = UniversalEngine(yaml_path=yaml_path, attention_type="sdpa")
out = engine.run(
    prompt=prompt,
    negative_prompt="",
    width=1536,
    height=1024,
    guidance_scale=4.0
)

out[0].save("test_ovis_background.png")
