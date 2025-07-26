from src.engine import create_engine
from diffusers.utils import export_to_video
import torch

engine = create_engine("stepvideo", "manifest/stepvideo/stepvideo_t2v_30b.yml", "t2v", save_path="/workspace/models", attention_type="sdpa", component_dtypes={"text_encoder": torch.float32})

prompt = "A beautiful woman in a flowing red dress walks through a dimly lit city street, neon lights reflecting off wet pavement, her silhouette glowing as the camera slowly tracks her from behind."
negative_prompt="画面暗、低分辨率、不良手、文本、缺少手指、多余的手指、裁剪、低质量、颗粒状、签名、水印、用户名、模糊。"

video = engine.run(
    prompt=prompt,
    negative_prompt=negative_prompt,
    height=544,
    width=992,
    duration=51,
    num_videos=1,
    num_inference_steps=30,
    guidance_scale=9.0,
)
export_to_video(video[0], "test_stepvideo_t2v_woman1.mp4", fps=25, quality=8)