from dotenv import load_dotenv
import torch
load_dotenv()
from src.engine.registry import UniversalEngine

yaml_path = "/home/tosin_coverquick_co/apex/manifest/engine/zimage/zimage-turbo-control-1.0.0.v1.yml"
engine = UniversalEngine(yaml_path=yaml_path)

prompt = """A man in his early thirties with a solid, well-balanced build and a calm, grounded presence. His head is slightly broad with a smooth, rounded crown. He has short, closely cropped dark hair, kept neat along the sides and back, with a subtle natural wave on top. His eyebrows are thick but tidy, framing deep-set eyes of a dark brown color that give him a steady, observant look. His eyes are relaxed rather than sharp, conveying quiet confidence.

His nose is straight and prominent, medium in width, with a softly rounded tip. His cheeks are full but firm, suggesting strength rather than softness. He has a strong, rounded jawline with a solid chin, not sharply angular but clearly defined. His lips are medium-full, naturally shaped, usually resting in a neutral expression.

His skin tone is medium to warm brown, even and smooth, with subtle natural shading around the eyes and jaw. Light stubble lines his jaw and upper lip, giving him a slightly rugged, mature appearance. His ears are average in size and sit close to the head. Overall, his features feel natural, masculine, and composed—someone who appears dependable, thoughtful, and quietly self-assured."""
control_image = "/home/tosin_coverquick_co/apex/assets/man.png"
out = engine.run(
    control_image=control_image,
    height=1088,
    width=720,
    prompt=prompt,
    seed=43,
    control_context_scale=0.8
)

out[0].save("output_zimage_control.png")