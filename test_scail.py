from dotenv import load_dotenv
load_dotenv()
from src.engine import UniversalEngine
from diffusers.utils import export_to_video

yaml_path = "/home/tosin_coverquick_co/apex/manifest/verified/video/wan-2.1-14b-scail-1.0.0.v1.yml"
engine = UniversalEngine(yaml_path=yaml_path, selected_components={
        "transformer": {
            "variant": "FP8",
            "precision": "fp8"
        }
    })

out = engine.run(
    image="/home/tosin_coverquick_co/apex/lord-farquaad.jpg",
    pose_video="/home/tosin_coverquick_co/apex/rendered_dance_dwpose_nlf.mp4",
    prompt="The man walks forwards and dances passionately with full raw and intense passion and energy, with a confident and assertive body language and over expressive facial expressions.",
    duration=81,
    height=896,
    width=512,
    num_inference_steps=4,
    guidance_scale=1.0,
)

export_to_video(out[0], "output_scail_hd.mp4", fps=16)