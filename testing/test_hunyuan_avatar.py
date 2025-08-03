from src.engine import create_engine
from diffusers.utils import export_to_video
import torch
from PIL import Image
import os 

engine = create_engine("hunyuan", "manifest/hunyuan/hunyuan_avatar.yml", "avatar", save_path="/workspace/models", attention_type="flash", component_dtypes={"text_encoder": torch.float32}, components_to_load=["transformer"], vae_tiling=True)

prompt= "A person with long blonde hair wearing a green jacket, standing in a forested area during twilight.",
negative_prompt="bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
image = Image.open("/workspace/HunyuanVideo-Avatar/assets/image/2.png")
audio = "/workspace/HunyuanVideo-Avatar/assets/audio/2.WAV"

video = engine.run(
    audio=audio,
    image=image,
    prompt=prompt,
    negative_prompt=negative_prompt,
    image_size=704,
    duration=129,
    num_videos=1,
    seed=42,
    num_inference_steps=50,
    guidance_scale=7.5
)

outfile_path = "base.mp4"
export_to_video(video[0], outfile_path, fps=25, quality=10)

os.system('ffmpeg -y -i /workspace/apex/base.mp4 -i "/workspace/HunyuanVideo-Avatar/assets/audio/2.WAV" -c:v libx264 -c:a aac -shortest /workspace/apex/hunyuan_avatar.mp4')
