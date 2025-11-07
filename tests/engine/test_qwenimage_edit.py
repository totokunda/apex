import os
from src.engine import create_engine
import json  
import torch

inputs = json.load(open("inputs.json", "r"))
engine = create_engine("qwenimage", "qwenimage-edit-1-0-0-v1", "edit", attention_type="sdpa")

images = engine.run(
    image=inputs["image"],
    prompt=inputs["prompt"],
    generator=torch.manual_seed(inputs["seed"]),
    true_cfg_scale=inputs["true_cfg_scale"],
    height=1024, 
    width=1024,
    negative_prompt=inputs["negative_prompt"],
    num_inference_steps=inputs["num_inference_steps"],
)

images[0].save("test_qwenimage_edit_1_0_0_couple+seed.png")   