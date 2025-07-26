from src.engine import create_engine
from diffusers.utils import export_to_video, load_image
import torch


engine = create_engine("stepvideo", "manifest/stepvideo/stepvideo_i2v_30b.yml", "i2v", save_path="/workspace/models", attention_type="sdpa", component_dtypes={"text_encoder": torch.float32})

prompt = "A solitary guitarist with cropped silver‑gray hair stands center stage, fingers poised on the glossy crimson neck of his electric guitar as soft blue and amber spotlights weave through drifting smoke, the polished body catching subtle lens flares and echoing the quiet intensity etched on his focused expression."
image = '/workspace/apex/guitar-man.png'
negative_prompt="画面暗、低分辨率、不良手、文本、缺少手指、多余的手指、裁剪、低质量、颗粒状、签名、水印、用户名、模糊。"

video = engine.run(
    image=image,
    prompt=prompt,
    negative_prompt=negative_prompt,
    height=480,
    width=832,
    duration=51,
    num_videos=1,
    num_inference_steps=50,
    seed=42,
    guidance_scale=9.0,
    motion_score=4.0,
)
export_to_video(video[0], "test_stepvideo_i2v_guitar.mp4", fps=25, quality=8)