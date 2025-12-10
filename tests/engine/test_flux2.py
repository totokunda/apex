from dotenv import load_dotenv
load_dotenv()
from diffusers.utils import export_to_video
from src.engine.registry import UniversalEngine

yaml_path = "/home/tosin_coverquick_co/apex/manifest/engine/flux2/flux2-dev-ti2i-1.0.0.v1.yml"
prompt = """On a wet street corner in a cyberpunk city at night, a large neon sign reading "I LOVE GINA" lights up sequentially, illuminating the dark, rainy environment with a pinkish-purple glow. The scene is a dark, rain-slicked street corner in a futuristic, cinematic cyberpunk city. Mounted on the metallic, weathered facade of a building is a massive, unlit neon sign. The sign's glass tube framework clearly spells out the words "I LOVE GINA". Initially, the street is dimly lit, with ambient light from distant skyscrapers creating shimmering reflections on the wet asphalt below. Then, the camera zooms in slowly toward the sign. As it moves, a low electrical sizzling sound begins. In the background, the dense urban landscape of the cyberpunk metropolis is visible through a light atmospheric haze, with towering structures adorned with their own flickering advertisements. A complex web of cables and pipes crisscrosses between the buildings. The shot is at a low angle, looking up at the sign to emphasize its grand scale. The lighting is high-contrast and dramatic, dominated by the neon glow which creates sharp, specular reflections and deep shadows. The atmosphere is moody and tech-noir. The overall video presents a cinematic photography realistic style."""
engine = UniversalEngine(yaml_path=yaml_path, attention_type="sdpa")
out = engine.run(
    prompt=prompt,
    guidance_scale=4.0,
    seed=42
)

out[0].save("test_flux2_i_love_gina_seed_42.png")
