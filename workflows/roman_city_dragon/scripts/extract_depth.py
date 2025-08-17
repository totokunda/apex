import sys
import os
os.makedirs("assets/depth", exist_ok=True)
os.environ["APEX_HOME_DIR"] = "/data"
sys.path.append('/data/apex')
from diffusers.utils import export_to_video

from src.preprocess.depth import VideoDepthAnythingV2Preprocessor


video = "/data/apex/workflows/roman_city_dragon/assets/flux_kontext/dragon_flux_kontext_continuation_1_cut.mp4"

preprocessor = VideoDepthAnythingV2Preprocessor()

depth = preprocessor(video)

export_to_video(depth.depth, "assets/depth/depth_dragon_flux_kontext_continuation_1_cut.mp4", fps=16, quality=8)