from dotenv import load_dotenv
load_dotenv()
from diffusers.utils import export_to_video
from src.engine.registry import UniversalEngine
import torch

yaml_path = "/home/divineade/apex/manifest/engine/wan/wan-2.2-14b-animate-1.0.0.v1.yml"
engine = UniversalEngine(yaml_path=yaml_path, attention_type="sdpa")



out = engine.run(
    image='/home/ext_diviade_gmail_com/apex-diffusion/cache/engine_results/9af76f08-67bc-4500-b35c-3cd1033e0d61/result.png',
    pose_video='/home/ext_diviade_gmail_com/apex-diffusion/cache/uploads/ea58fad6-d8c2-41dc-8466-fc9224ccd933-eda93808-c581-4a59-a306-3ff3785ae7e4_pose_video_0_120.mp4',
    face_video='/home/ext_diviade_gmail_com/apex-diffusion/cache/uploads/88d1e29f-2915-4415-a955-e2445be98d7d-eda93808-c581-4a59-a306-3ff3785ae7e4_face_video_0_120.mp4',
    duration=81,
    height=832,
    width=480,
    guidance_scale=1.0,
    prompt="A dancing in an alley while its raining",
    num_inference_steps=50,
    generator=torch.Generator(device="cuda").manual_seed(490804809),
    mode="animate",
)

export_to_video(out[0], "animated_character_test_1.mp4", fps=16)