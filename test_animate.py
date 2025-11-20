from dotenv import load_dotenv
load_dotenv()
from diffusers.utils import export_to_video
from src.engine.registry import UniversalEngine
import torch

yaml_path = "/home/tosin_coverquick_co/apex/manifest/engine/wan/wan-2.2-14b-animate-1.0.0.v1.yml"
engine = UniversalEngine(yaml_path=yaml_path, attention_type="sdpa")

out = engine.run(
    image='/home/tosin_coverquick_co/apex/process_results/src_ref.png',
    pose_video='/home/tosin_coverquick_co/apex/process_results/src_pose.mp4',
    face_video='/home/tosin_coverquick_co/apex/process_results/src_face.mp4',
    duration=81,
    height=480,
    width=832,
    guidance_scale=1.0,
    prompt="The man energetically and frantically moves his hands as he speaks passionately.",
    num_inference_steps=20,
    generator=torch.Generator(device="cuda").manual_seed(42),
    mode="animate",
)

export_to_video(out[0], "animated_character_test.mp4", fps=16)