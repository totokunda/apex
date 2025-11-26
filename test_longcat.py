from dotenv import load_dotenv
load_dotenv()
from diffusers.utils import export_to_video
from src.engine.registry import UniversalEngine

yaml_path = "/home/tosin_coverquick_co/apex/manifest/engine/longcat/longcat-13b-text-to-video-1.0.0.v1.yml"
engine = UniversalEngine(yaml_path=yaml_path, attention_type="flash")

prompt = """On a wet street corner in a cyberpunk city at night, a large neon sign reading "I LOVE GINA" lights up sequentially, illuminating the dark, rainy environment with a pinkish-purple glow. The scene is a dark, rain-slicked street corner in a futuristic, cinematic cyberpunk city. Mounted on the metallic, weathered facade of a building is a massive, unlit neon sign. The sign's glass tube framework clearly spells out the words "I LOVE GINA". Initially, the street is dimly lit, with ambient light from distant skyscrapers creating shimmering reflections on the wet asphalt below. Then, the camera zooms in slowly toward the sign. As it moves, a low electrical sizzling sound begins. In the background, the dense urban landscape of the cyberpunk metropolis is visible through a light atmospheric haze, with towering structures adorned with their own flickering advertisements. A complex web of cables and pipes crisscrosses between the buildings. The shot is at a low angle, looking up at the sign to emphasize its grand scale. The lighting is high-contrast and dramatic, dominated by the neon glow which creates sharp, specular reflections and deep shadows. The atmosphere is moody and tech-noir. The overall video presents a cinematic photography realistic style."""

out = engine.run(
    prompt=prompt,
    use_distill=True,
    num_inference_steps=16,
    duration=93,
    guidance_scale=1.0,
    enable_refine=True,
)

export_to_video(out[0], "output_longcat_long_video_img.mp4", fps=30)