from dotenv import load_dotenv
load_dotenv()
from diffusers.utils import export_to_video
from src.engine.registry import UniversalEngine

yaml_path = "/home/tosin_coverquick_co/apex/manifest/engine/wan/krea-realtime-a14b-text-to-video-1.0.0.v1.yml"
prompt = "A dog running in a park with a ball in its mouth."

engine = UniversalEngine(yaml_path=yaml_path, attention_type="sdpa")
out = engine.run(
    prompt=prompt,
    duration=81,
    height=480,
    width=832
)

export_to_video(out[0], "test_krea.mp4", fps=24, quality=8)
