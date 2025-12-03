from dotenv import load_dotenv
import torch
load_dotenv()
from src.engine.registry import UniversalEngine

yaml_path = "/home/tosin_coverquick_co/apex/manifest/engine/flux2/flux2-dev-control-1.0.0.v1.yml"
engine = UniversalEngine(yaml_path=yaml_path)

prompt = "This is a panoramic portrait photo of a young woman. She has flowing long hair and a soft lavender like color. She is wearing a white sleeveless dress with a blue ribbon bow tied around the collar. She has a confident posture, with her left hand naturally hanging down and her right hand in her pocket, and her legs slightly apart. Look straight at the camera. The sea breeze gently brushed her long hair, and they stood on the sunny seaside path, surrounded by blooming purple seaside flowers and smooth pebbles, with the sparkling sea and blue sky behind them. The screen presents a bright summer atmosphere, with soft and natural lighting, realistic details, and 8K ultra high definition image quality, clearly presenting fine textures such as clothing and hair. "
control_image = "/home/tosin_coverquick_co/apex/VideoX-Fun/asset/pose.jpg"
out = engine.run(
    control_image=control_image,
    prompt=prompt,
    seed=43,
    guidance_scale=4.0,
    control_context_scale=0.75
)

out[0].save("output_flux2_control.png")