from dotenv import load_dotenv
load_dotenv()
from src.engine.registry import UniversalEngine
import json 
from diffusers.utils import export_to_video
import numpy as np
from typing import Optional
import tempfile
import soundfile as wavfile
import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

with open("/home/divineade/apex/runs/wan-2.2-14b-animate-1.0.0.v1/model_inputs.json", "r") as f:
   data = json.load(f)

engine_kwargs = data["engine_kwargs"]

print(engine_kwargs)

prompt = "A cinematic space-themed scene set against a deep starfield and distant galaxies. A small group of people in sleek, futuristic attire stand together on a glowing platform floating in space, gazing outward with a sense of creativity and ambition. Above and slightly behind them, the text “Apex Studio” appears large and clearly readable, styled in modern, bold typography with a soft luminous glow. Dramatic lighting, subtle nebula colors, depth and scale, high detail but clean composition, professional sci-fi creative aesthetic."

inputs = data["inputs"]
base_path = "/home/divineade/apex/runs/wan-2.2-14b-animate-1.0.0.v1"
inputs["pose_video"] = os.path.join(base_path, inputs["pose_video"])
inputs["face_video"] = os.path.join(base_path, inputs["face_video"])
inputs["image"] = os.path.join(base_path, inputs["image"])

# inputs["prompt"] = prompt


yaml_path = engine_kwargs.get("yaml_path")
engine = UniversalEngine(yaml_path=yaml_path, selected_components = {

            "transformer": {
                "path": "/home/ext_diviade_gmail_com/apex-diffusion/components/4d070faacff0a3bfc145235c1eb9432a1451c06f5be5da155a6d51fe16c58348_Wan2_2-Animate-14B_fp8_scaled_e4m3fn_KJ_v2.safetensors",
                "variant": "FP8",
                "precision": "fp8",
                "type": "safetensors"
            },
    
})

print(inputs)
out = engine.run(
    **inputs
)

# base_path = "/home/divineade/apex/assets/demo_images"
# model_name = "qwen_image_edit.jpg"
# output_path = os.path.join(base_path, model_name)

# print(output_path)

# out[0].save("output.png")

export_to_video(out[0], "output_16.mp4", fps=16)
# export_to_video(out[0], "output_24.mp4", fps=24)

