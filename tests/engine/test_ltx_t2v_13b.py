import os

from cv2.gapi import video
os.environ["APEX_HOME_DIR"] = "/mnt/localssd"
from src.engine import create_engine
from diffusers.utils import export_to_video
from src.engine.ltx.x2v import LTXVideoCondition
from src.engine.ltx.base import calculate_padding
from src.postprocess.ltx import LatentUpsamplerPostprocessor
from PIL import Image
from typing import List
import torch

engine = create_engine("ltx", "ltx-x2v-13b-0-9-8-dev-1-0-0-v1", "x2v", attention_type="sdpa", components_to_load=['vae'], component_load_dtypes={"vae": torch.float32}, component_dtypes={"vae": torch.bfloat16, "text_encoder": torch.bfloat16, "transformer": torch.bfloat16})
postprocessor = LatentUpsamplerPostprocessor(engine, dtype=torch.float32,
                                             model_path="https://huggingface.co/Lightricks/LTX-Video/resolve/main/ltxv-spatial-upscaler-0.9.8.safetensors",
                                             config={"_class_name": "LatentUpsampler", "in_channels": 128, "mid_channels": 512, "num_blocks_per_stage": 4, "dims": 3})

image = Image.open('/home/tosinkuye/apex/assets/image/dog.png')

height = 704
width = 1216
height_padded = ((height - 1) // 32 + 1) * 32
width_padded = ((width - 1) // 32 + 1) * 32
padding = calculate_padding(height, width, height_padded, width_padded)

condition = LTXVideoCondition(
    image=image,
    padding=padding
)

prompt = "A playful golden retriever puppy sitting in a grassy field with scattered orange and yellow flowers, the camera slowly pans in while the puppy happily looks around and wags its tail, ears perking up as it moves its head curiously, natural daylight with a soft cinematic look."
negative_prompt="lowres, text, error, cropped, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, username, watermark, signature"

downscale_factor = 2/3
x_width = int(width * downscale_factor)
downscaled_width = x_width - (x_width % engine.vae_scale_factor_spatial)
x_height = int(height * downscale_factor)
downscaled_height = x_height - (x_height % engine.vae_scale_factor_spatial)
generator = torch.Generator(device="cuda").manual_seed(42)

output = engine.run(
    conditions=[condition],
    prompt=prompt,
    negative_prompt=negative_prompt,
    height=downscaled_height,
    width=downscaled_width,
    duration=121,
    num_videos=1,
    num_inference_steps=30,
    skip_final_inference_steps=3,
    generator=generator,
    return_latents=False 
)

upsampled_latents = postprocessor(video=output[0], return_latents=True)

upsampled_video: List[Image.Image] = engine.run(
    conditions=[condition],
    prompt=prompt,
    negative_prompt=negative_prompt,
    initial_latents=upsampled_latents,
    height=downscaled_height*2,
    width=downscaled_width*2,
    duration=121,
    num_videos=1,
    num_inference_steps=30,
    guidance_scale=[1],
    stg_scale=[1],
    rescaling_scale=[1],
    guidance_timesteps=[1.0],
    cfg_star_rescale=True,
    skip_initial_inference_steps=17,
    skip_final_inference_steps=0,
    generator=generator
)[0]

for i, frame in enumerate(upsampled_video):
    upsampled_video[i] = frame.resize((width, height), Image.Resampling.LANCZOS)

export_to_video(upsampled_video, "test_ltx_t2v_13b_dog_upsampled.mp4", fps=30, quality=8)