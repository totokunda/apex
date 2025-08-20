from src.postprocess.rife import RifePostprocessor  
from diffusers.utils import export_to_video

postprocessor = RifePostprocessor()

video = "apex/workflows/roman_city_dragon/assets/flux_kontext/dragon_flux_kontext.mp4"

video = postprocessor(video, multi=2)

export_to_video(video, "test_rife.mp4", fps=32, quality=9)