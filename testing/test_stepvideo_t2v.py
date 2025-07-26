from src.engine import create_engine
from diffusers.utils import export_to_video
import torch

engine = create_engine("stepvideo", "manifest/stepvideo/stepvideo_t2v_13b.yml", "t2v", save_path="/dev/shm/models", attention_type="sdpa", component_dtypes={"text_encoder": torch.float32})

prompt = "A beautiful woman in a flowing red dress walks through a dimly lit city street, neon lights reflecting off wet pavement, her silhouette glowing as the camera slowly tracks her from behind."
negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"

video = engine.run(
    prompt=prompt,
    negative_prompt=negative_prompt,
    height=384,
    width=640,
    duration='34f',
    num_videos=1,
    num_inference_steps=50,
    guidance_scale=9.0,
    generator=torch.Generator(device="cuda").manual_seed(69)
)
export_to_video(video[0], "test_stepvideo_t2v_woman1.mp4", fps=25, quality=8)