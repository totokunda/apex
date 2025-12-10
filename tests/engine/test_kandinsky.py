from dotenv import load_dotenv
load_dotenv()
from diffusers.utils import export_to_video
from src.engine.registry import UniversalEngine

yaml_path = "/home/tosin_coverquick_co/apex/manifest/engine/kandinsky/kandinsky-i2v-pro-5s-1.0.0.v1.yml"
engine = UniversalEngine(yaml_path=yaml_path)
image = "/home/tosin_coverquick_co/apex/images/apex_woman_image.png"
prompt = "The woman takes the sign she is holding and throws it on the ground, angrily, before walking away."

negative_prompt = ""
out = engine.run(
    image=image,
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=25,
    height=768,
    width=512,
    num_frames=121,
    guidance_scale=5.0,
)
export_to_video(out[0], "output_kandinsky_i2v_pro_5s.mp4", fps=24)