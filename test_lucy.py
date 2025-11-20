from dotenv import load_dotenv
load_dotenv()
from src.engine.registry import UniversalEngine
from diffusers.utils import export_to_video
yaml_path = "/home/tosin_coverquick_co/apex/manifest/engine/lucy/lucy-edit-1.1-dev-1.0.0.v1.yml"
engine = UniversalEngine(yaml_path=yaml_path)

out = engine.run(
    video='/home/tosin_coverquick_co/apex/whipping_pose.mp4',
    prompt="Make the man have dark skin.",
    num_inference_steps=30,
    guidance_scale=5.0,
    num_frames=81,
)

export_to_video(out[0], "output_lucy.mp4", fps=24)