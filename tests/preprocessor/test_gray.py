from src.preprocess.gray import GrayPreprocessor, GrayVideoPreprocessor

from diffusers.utils import export_to_video
from PIL import Image

image = "assets/image/dog.png"
video = "assets/video/conversation.mp4"


preprocessor = GrayVideoPreprocessor()
output = preprocessor(video)

export_to_video(output.frames, "assets/test/gray.mp4", fps=24)

preprocessor = GrayPreprocessor()
output = preprocessor(image)
output.frame.save("assets/test/gray.png")