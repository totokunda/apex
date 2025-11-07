import os
from PIL import Image
import torch
import json
from diffusers import QwenImageEditPipeline

pipeline = QwenImageEditPipeline.from_pretrained("Qwen/Qwen-Image-Edit")
print("pipeline loaded")
pipeline.to(torch.bfloat16)
pipeline.to("cuda")
pipeline.set_progress_bar_config(disable=None)
inputs = json.load(open("inputs.json", "r"))

image = Image.open(inputs["image"]).convert("RGB")
prompt = inputs["prompt"]
inputs = {
    "image": image,
    "prompt": prompt,
    "generator": torch.manual_seed(inputs["seed"]),
    "true_cfg_scale": inputs["true_cfg_scale"],
    "negative_prompt": inputs["negative_prompt"],
    "num_inference_steps": inputs["num_inference_steps"],
}

with torch.inference_mode():
    output = pipeline(**inputs)
    output_image = output.images[0]
    output_image.save("output_image_edit.png")
    print("image saved at", os.path.abspath("output_image_edit.png"))
