from src.postprocess.rife import RifePostprocessor
from diffusers.utils.export_utils import export_to_video

rife = RifePostprocessor(target_fps=60)

input_video = '/Users/tosinkuye/Library/Application Support/apex-studio/media/symlinks/89b2f25e1a444ac5bbb22fac6df387bd.mp4'


output = rife(
    video=input_video,
)

export_to_video(output, 'output.mp4', fps=60)