from transformers import Qwen2_5_VLForConditionalGeneration, PretrainedConfig
import json
import os
path = "/home/tosin_coverquick_co/apex-diffusion/components/Qwen_Qwen-Image/text_encoder"

with open(os.path.join(path, "config.json"), "r") as f:
    config = json.load(f)

config = PretrainedConfig.from_dict(config)
print(config)