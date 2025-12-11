from dotenv import load_dotenv
import torch
load_dotenv()
from src.engine.registry import UniversalEngine
from diffusers.utils import export_to_video
yaml_path = "/home/tosin_coverquick_co/apex/manifest/engine/lucy/lucy-edit-1.1-dev-1.0.0.v1.yml"
engine = UniversalEngine(yaml_path=yaml_path, component_dtypes={
    "vae": torch.float32,
})

out = engine.run(
    video='/home/tosin_coverquick_co/apex/man_jacket_snow_edit.mp4',
    prompt="Turn the man into a bear.",
    negative_prompt="lowres, text, error, cropped, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, username, watermark, signature",
    num_inference_steps=50,
    height=720,
    width=1280,
    guidance_scale=5.0,
    num_frames=81,
    generator=torch.Generator(device="cuda").manual_seed(42),
)

export_to_video(out[0], "output_lucy.mp4", fps=24)