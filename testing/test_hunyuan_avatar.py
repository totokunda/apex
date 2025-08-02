from src.engine import create_engine
from diffusers.utils import export_to_video
import torch
from PIL import Image

engine = create_engine("hunyuan", "manifest/hunyuan/hunyuan_avatar.yml", "avatar", save_path="/workspace/models", attention_type="sdpa", component_dtypes={"text_encoder": torch.float32}, components_to_load=["transformer"])

prompt= "In a cozy recording studio, a man and a woman are singing together with passion and emotion. The man, with short brown hair, wears a light gray button-up shirt, his expression filled with concentration and warmth. The woman, with long wavy brown hair, dons a sleeveless dress adorned with small polka dots, her eyes closed as she belts out a heartfelt melody. The studio is equipped with professional microphones, and the background features soundproofing panels, creating an intimate and focused atmosphere. A close-up shot captures their expressions and the intensity of their performance.",
negative_prompt="bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
image = Image.open("/workspace/MultiTalk/examples/multi/2/multi2.png")

video = engine.run(
    audio= "/workspace/MultiTalk/examples/multi/2/1.wav",
    image=image,
    prompt=prompt,
    negative_prompt=negative_prompt,
    height=448,
    width=896,
    duration=129,
    num_videos=1,
    num_inference_steps=40,
    guidance_scale=5.0,
    dynamic_guidance_start=3.5,
    dynamic_guidance_end=6.5,
    seed=42
)

outfile_path = "test_wan_avatar.mp4"
export_to_video(video[0], outfile_path, fps=25, quality=5)
