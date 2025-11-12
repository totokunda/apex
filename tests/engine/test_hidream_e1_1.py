import os
os.environ["APEX_HOME_DIR"] = "/mnt/localssd"
os.environ['HF_HOME'] = '/mnt/localssd'
from src.engine import create_engine
from PIL import Image

engine = create_engine("hidream", "hidream-e1-1-1-0-0-v1", "edit", attention_type="sdpa")

editing_instructions = "Convert the image into a Ghibli style."
negative_prompt="low quality, blurry, distorted"

image = Image.open("/home/tosinkuye/apex/doog_rgb.png")
og_size = image.size

video = engine.run(
    image=image,
    prompt=editing_instructions,
    negative_prompt=negative_prompt,
    num_images=1,
    num_inference_steps=28,
    guidance_scale=3.0,
    image_guidance_scale=1.5,
    refine_strength=0.3,
    seed=3
)

video[0][0].save("test_hidream_e1_full_edit_couple_1_0_0_doog.png")   
#export_to_video(video[0], "test_flux_kontext_1_0_0_dog.mp4", fps=16, quality=8)