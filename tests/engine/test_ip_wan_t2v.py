import os 
os.environ["APEX_HOME_DIR"] = "/mnt/localssd"
from src.engine import create_engine
from diffusers.utils import export_to_video
from src.preprocess.face import StandinFacePreprocessor
import torch

engine = create_engine("wan", "manifest/wan/wan_ip_t2v_14b.yml", "t2v", attention_type="flash", save_converted_weights=False)

prompt="A man sits comfortably at a desk, facing the camera as if talking to a friend or family member on the screen. His gaze is focused and gentle, with a natural smile. The background is his carefully decorated personal space, with photos and a world map on the wall, conveying a sense of intimate and modern communication." 
negative_prompt="lowres, text, error, cropped, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, username, watermark, signature"
face_preprocessor = StandinFacePreprocessor()
face = face_preprocessor("Stand-In/test/input/lecun.jpg").face

video = engine.run(
    prompt=prompt,
    negative_prompt=negative_prompt,
    height=480,
    width=832,
    duration=81,
    num_videos=1,
    num_inference_steps=20,
    guidance_scale=5.0, 
    seed=0,
    ip_image=face,
)

export_to_video(video[0], "test_wan_ip_t2v_14b.mp4", fps=25, quality=8)