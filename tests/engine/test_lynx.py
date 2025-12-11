from dotenv import load_dotenv
load_dotenv()
from diffusers.utils import export_to_video
from src.engine.registry import UniversalEngine
yaml_path = "/home/tosin_coverquick_co/apex/manifest/engine/wan/lynx-14b-1.0.0.v1.yml"

engine = UniversalEngine(yaml_path=yaml_path, attention_type="sdpa")
out = engine.run(
    subject_image='/home/tosin_coverquick_co/apex/IMG_7555.jpg',
    prompt="The man in the video takes off his shirt, slowly and sensually as he looks at the camera.",
    duration=81,
    height=480,
    width=832,
    negative_prompt="lowres, text, error, cropped, worst quality, low quality, jpeg artifacts, \
        ugly, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, \
        poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, \
        bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross \
        proportions, malformed limbs, missing arms, missing legs, extra arms, extra \
        legs, fused fingers, too many fingers, long neck, username, watermark, signature",
    num_inference_steps=50,
    seed=69
)

export_to_video(out[0], "test_lynx.mp4", fps=16, quality=8)
