import os 
os.environ["APEX_HOME_DIR"] = "/mnt/localssd"

from src.engine import create_engine
from diffusers.utils import export_to_video
import torch

engine = create_engine("wan", "manifest/wan/wan_recam_1.3b.yml", "recam", attention_type="flash",  save_converted_weights=False)
camera_extrinsics = 'assets/wan/recam_extrinsics.json'

prompt = "A man and a woman are dancing together on a city street at dusk. The woman is wearing a yellow dress with a floral pattern and black shoes, while the man is dressed in a white shirt, dark tie, and dark pants with black shoes. They are both moving gracefully to the rhythm of the music, with the woman twirling and the man following her steps. The background features a cityscape with buildings and a distant mountain range, illuminated by the setting sun, creating a warm and romantic atmosphere. The main subjects are a man and a woman. The woman is wearing a yellow dress with a floral pattern and black shoes, while the man is dressed in a white shirt, dark tie, and dark pants with black shoes. They are dancing closely together, with the woman often twirling and the man following her movements. Their expressions and body language suggest they are enjoying the dance and are in sync with each other. The woman's movements are fluid and expressive, involving twirls and spins. The man follows her lead, moving in sync with her steps. Their dance is characterized by moderate to fast-paced movements, with the woman often leading the way. The background remains static, emphasizing the dynamic nature of their dance."
negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"

video_path = 'ReCamMaster/example_test_data/videos/1.mp4'

video = engine.run(
    source_video=video_path,
    camera_extrinsics=camera_extrinsics,
    prompt=prompt,
    negative_prompt=negative_prompt,
    height=480,
    width=832,
    duration=81,
    num_videos=1,
    num_inference_steps=50,
    guidance_scale=5.0
)

export_to_video(video[0], "test_wan_recam_1_3b.mp4", fps=30, quality=8)