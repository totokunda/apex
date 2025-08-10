from src.engine import create_engine
from diffusers.utils import export_to_video
import torch
from PIL import Image

engine = create_engine("wan", "manifest/wan/wan_22_ti2v_5b.yml", "i2v", save_path="/workspace/models", attention_type="flash", vae_scale_factor=16, component_dtypes={"vae": torch.float32})

prompt="In a cozy recording studio, a man and a woman are singing together with passion and emotion. The man, with short brown hair, wears a light gray button-up shirt, his expression filled with concentration and warmth. The woman, with long wavy brown hair, dons a sleeveless dress adorned with small polka dots, her eyes closed as she belts out a heartfelt melody. The studio is equipped with professional microphones, and the background features soundproofing panels, creating an intimate and focused atmosphere. A close-up shot captures their expressions and the intensity of their performance."
negative_prompt="lowres, text, error, cropped, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, username, watermark, signature"
image = Image.open("/workspace/MultiTalk/examples/multi/2/multi2.png")

video = engine.run(
    image=image,
    prompt=prompt,
    negative_prompt=negative_prompt,
    height=704,
    width=1280,
    duration=121,
    num_videos=1,
    num_inference_steps=50,
    guidance_scale=5.0
)

export_to_video(video[0], "test_wan_22_ti2v_i2v.mp4", fps=24, quality=8)