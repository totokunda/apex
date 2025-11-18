import numpy as np
import torch
from diffusers import AutoencoderKLWan, WanAnimatePipeline
from diffusers.utils import export_to_video, load_image, load_video
from transformers import CLIPVisionModel

model_id = "Wan-AI/Wan2.2-Animate-14B-Diffusers"
vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
pipe = WanAnimatePipeline.from_pretrained(
    model_id, vae=vae, torch_dtype=torch.bfloat16
)
pipe.to("cuda")

# Load character image and preprocessed videos
image = load_image("/home/tosin_coverquick_co/apex/process_results/src_ref.png")
pose_video = load_video("/home/tosin_coverquick_co/apex/process_results/src_pose.mp4")  # Preprocessed skeletal keypoints
face_video = load_video("/home/tosin_coverquick_co/apex/process_results/src_face.mp4")  # Preprocessed facial features

# Resize image to match VAE constraints
def aspect_ratio_resize(image, pipe, max_area=480 * 832):
    aspect_ratio = image.height / image.width
    mod_value = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
    height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
    width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
    image = image.resize((width, height))
    return image, height, width

image, height, width = aspect_ratio_resize(image, pipe)

prompt = "The man energetically and frantically moves his hands as he speaks passionately."
negative_prompt = "lowres, text, error, cropped, worst quality, low quality, jpeg artifacts, \
        ugly, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, \
        poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, \
        bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross \
        proportions, malformed limbs, missing arms, missing legs, extra arms, extra \
        legs, fused fingers, too many fingers, long neck, username, watermark, signature"

# Generate animated video
output = pipe(
    image=image,
    pose_video=pose_video,
    face_video=face_video,
    prompt=prompt,
    height=height,
    width=width,
    guidance_scale=1.0,
    generator=torch.Generator(device="cuda").manual_seed(42),
    mode="animate",  # Animation mode (default)
).frames[0]
export_to_video(output, "animated_character.mp4", fps=16)