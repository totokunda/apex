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

with open("/home/divineade/apex/src/runs/srpo-text-to-image-1.0.0.v1/model_inputs.json", "r") as f:
   data = json.load(f)


engine_kwargs = data["engine_kwargs"]

prompt = "A cinematic space-themed scene set against a deep starfield and distant galaxies. A small group of people in sleek, futuristic attire stand together on a glowing platform floating in space, gazing outward with a sense of creativity and ambition. Above and slightly behind them, the text “Apex Studio” appears large and clearly readable, styled in modern, bold typography with a soft luminous glow. Dramatic lighting, subtle nebula colors, depth and scale, high detail but clean composition, professional sci-fi creative aesthetic."

inputs = data["inputs"]





yaml_path = engine_kwargs.get("yaml_path")
engine = UniversalEngine(yaml_path=yaml_path, selected_components = {

    "transformer": {
                "path": "/home/ext_diviade_gmail_com/apex-diffusion/components/e27fa98ae4ea9da8e49be42f0f4828afa21cab6d38d264954edb898de68db5d4_srpo-Q4_K.gguf",
                "variant": "q4_k",
                "precision": "q4_k",
                "type": "gguf"
            }
})


out = engine.run(
    **inputs
)

# base_path = "/home/divineade/apex/assets/demo_images"
# model_name = "qwen_image_edit.jpg"
# output_path = os.path.join(base_path, model_name)

# print(output_path)

out[0].save("output.png")

# export_to_video(out[0], "output_16.mp4", fps=16)
# export_to_video(out[0], "output_24.mp4", fps=24)

