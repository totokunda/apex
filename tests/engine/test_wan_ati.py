import os 
os.environ['APEX_HOME_DIR'] = '/mnt/localssd'
from src.engine import create_engine
from diffusers.utils import export_to_video
import torch

engine = create_engine("wan", "manifest/wan/wan_ati_14b.yml", "ati", attention_type="flash", component_dtypes={"text_encoder": torch.float32}, save_converted_weights=False)

prompt = "A brown bear lying in the shade beside a rock, resting on a bed of grass."
negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
image = '/home/tosinkuye/apex/ATI/examples/images/bear.jpg'
trajectory = '/home/tosinkuye/apex/ATI/examples/tracks/bear.pth'
    
video = engine.run(
    image=image,
    trajectory=trajectory,
    prompt=prompt,
    negative_prompt=negative_prompt,
    height=480,
    width=832,
    duration=81,
    num_videos=1,
    num_inference_steps=40,
    guidance_scale=5.0
)

export_to_video(video[0], "test_wan_ati_bear.mp4", fps=16, quality=8)