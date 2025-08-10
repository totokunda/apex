from src.engine import create_engine
from diffusers.utils import export_to_video
import torch

engine = create_engine("cosmos2", "manifest/cosmos/cosmos2_x2v_2b.yml", "v2v", save_path="/workspace/models", attention_type="flash", component_dtypes={"text_encoder": torch.float32})

prompt = "The man shakes his head slowly at the camera then begins to burst into flames."
negative_prompt = "The video captures a series of frames showing ugly scenes, static with no motion, motion blur, over-saturation, shaky footage, low resolution, grainy texture, pixelated images, poorly lit areas, underexposed and overexposed scenes, poor color balance, washed out colors, choppy sequences, jerky movements, low frame rate, artifacting, color banding, unnatural transitions, outdated special effects, fake elements, unconvincing visuals, poorly edited content, jump cuts, visual noise, and flickering. Overall, the video is of poor quality."
video_path = '/workspace/apex/man.mp4'

video = engine.run(
    video=video_path,
    prompt=prompt,
    negative_prompt=negative_prompt,
    height=704,
    width=1280,
    duration=93,
    num_videos=1,
    num_inference_steps=35,
    guidance_scale=7.0
)

export_to_video(video[0], "test_cosmos2_x2v_2b_smile.mp4", fps=16, quality=8)