from src.preprocess.canny import CannyVideoPreprocessor
from diffusers.utils import export_to_video
video = "assets/video/conversation.mp4"

preprocessor = CannyVideoPreprocessor()
output = preprocessor(video)

export_to_video(output.frames, "assets/test/canny.mp4", fps=24)