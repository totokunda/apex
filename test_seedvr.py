from src.engine.registry import UniversalEngine
from diffusers.utils import export_to_video
yaml_path = "/home/tosin_coverquick_co/apex/manifest/verified/upscalers/seedvr2-7b.yml"
engine = UniversalEngine(yaml_path=yaml_path)

video = "/home/tosin_coverquick_co/apex/dragon_flux_kontext.mp4"
out = engine.run(
    video=video,
    height=1080,
    width=1080,
    num_inference_steps=1,
    seed=666,
    vae_conv_max_mem=0.1,
    vae_norm_max_mem=0.1,
    vae_split_size=4,
    vae_memory_device="same",
    chunk_frames=33,
    chunk_overlap=8,
)

export_to_video(out[0], "test_seedvr_7b_chunked.mp4", fps=24, quality=8)