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
import torch
import requests
import io
from huggingface_hub import get_token
torch.set_printoptions(threshold=10000, linewidth=300)

directory = "/home/tosin_coverquick_co/apex/runs/flux2-dev-text-to-image-edit-1.0.0.v1"

with open(os.path.join(directory, "model_inputs.json"), "r") as f:
   data = json.load(f)

engine_kwargs = data["engine_kwargs"]

inputs = data["inputs"]
for input_key, input_value in inputs.items():
    if isinstance(input_value, str) and input_value.startswith("assets"):
        inputs[input_key] = os.path.join(directory, input_value)

import time
start_time = time.perf_counter()
engine = UniversalEngine(**engine_kwargs)


out = engine.run(
    **inputs
)
out[0].save("output.png")
#export_to_video(out[0], "output.mp4", fps=16, quality=8)