from dotenv import load_dotenv
load_dotenv()
from src.engine.registry import UniversalEngine
import json 
from diffusers.utils import export_to_video

with open("inputs.json", "r") as f:
    data = json.load(f)

engine_kwargs = data["engine_kwargs"]
inputs = data["inputs"]

engine = UniversalEngine(**engine_kwargs)

out = engine.run(
    **{**inputs,
    "image": '/home/tosin_coverquick_co/apex/process_results/src_ref.png',
    "pose_video": '/home/tosin_coverquick_co/apex/process_results/src_pose.mp4',
    "face_video": '/home/tosin_coverquick_co/apex/process_results/src_face.mp4',
    "background_video": '/home/tosin_coverquick_co/apex/process_results/src_bg.mp4',
    "mask_video": '/home/tosin_coverquick_co/apex/process_results/src_mask.mp4',
    "mode": "replace",
    },
)

export_to_video(out[0], "output_test.mp4", fps=16)