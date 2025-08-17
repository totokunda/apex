import sys
import os
os.environ["APEX_HOME_DIR"] = "/data"
sys.path.append('/data/apex')
os.makedirs("assets/inpainting", exist_ok=True)

from src.preprocess.inpainting import InpaintingVideoPreprocessor
from diffusers.utils import export_to_video
from PIL import Image
video = "/data/apex/workflows/roman_city_dragon/assets/flux_kontext/dragon_flux_kontext.mp4"

preprocessor = InpaintingVideoPreprocessor()

out = preprocessor(frames=video, mode='salientmasktrack', return_mask=True)
frames = [Image.fromarray(frame) for frame in out.frames]
masks = [Image.fromarray(mask) for mask in out.masks]

export_to_video(frames, "assets/inpainting/dragon_flux_kontext_inpainting.mp4", fps=16, quality=9)
export_to_video(masks, "assets/inpainting/dragon_flux_kontext_inpainting_mask.mp4", fps=16, quality=9)
