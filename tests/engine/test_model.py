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


with open("demo_images_inputs.json", "r") as f:
   data = json.load(f)


engine_kwargs = data["engine_kwargs"]
inputs = data["inputs"]


yaml_path = engine_kwargs.get("yaml_path")
engine = UniversalEngine(yaml_path=yaml_path)

out = engine.run(
    **inputs
)

base_path = "/home/divineade/apex/assets/demo_images"
model_name = "qwen_image_edit.jpg"
output_path = os.path.join(base_path, model_name)

print(output_path)

out[0].save(output_path)

# export_to_video(out[0], "output_16.mp4", fps=16)
# export_to_video(out[0], "output_24.mp4", fps=24)

