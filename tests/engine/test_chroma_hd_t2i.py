import os
os.environ["APEX_HOME_DIR"] = "/mnt/localssd"
from src.engine import create_engine

engine = create_engine("chroma", "chroma-hd-text-to-image-1-0-0-v1", "t2i", attention_type="sdpa", components_to_load=['transformer'])

prompt = [
    "A high-fashion close-up portrait of a blonde woman in clear sunglasses. The image uses a bold teal and red color split for dramatic lighting. The background is a simple teal-green. The photo is sharp and well-composed, and is designed for viewing with anaglyph 3D glasses for optimal effect. It looks professionally done."
]
negative_prompt =  ["low quality, ugly, unfinished, out of focus, deformed, disfigure, blurry, smudged, restricted palette, flat colors"]

video = engine.run(
    prompt=prompt,
    negative_prompt=negative_prompt,
    height=1024,
    width=1024,
    num_images=1,
    num_inference_steps=40,
    guidance_scale=3.5
)

video[0][0].save("test_chroma_hd_t2i_1_0_0_glasses.png") 
#export_to_video(video[0], "test_flux_t2i_1_0_0_cat.mp4", fps=16, quality=10)