from src.engine.registry import UniversalEngine
from diffusers.utils import export_to_video

yaml_path = "/home/tosin_coverquick_co/apex/manifest/verified/upscalers/flashvsr-1.1-full.yml"

engine = UniversalEngine(yaml_path=yaml_path)

video = "/home/tosin_coverquick_co/apex/christimas_anmate_1.mov"
out = engine.run(
    video=video,
    scale_factor=3,
    buffer=True,
    color_fix=True,
    sparse_ratio=2.0,
    local_range=11,
    kv_ratio=3.0,
)

#out[0].save("test_flashvsr.png")
export_to_video(out[0], "killing_it_with_christmas_full.mp4", fps=24, quality=7)