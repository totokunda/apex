from src.preprocess.scribble import ScribblePreprocessor, ScribbleVideoPreprocessor
from diffusers.utils import export_to_video
from PIL import Image
import os


image = "assets/image/dog.png"
video = "assets/video/conversation.mp4"

test_dir = "assets/test"
os.makedirs(test_dir, exist_ok=True)


print("Testing ScribblePreprocessor")
pre = ScribblePreprocessor()
out = pre(image)
out.image.save(os.path.join(test_dir, "scribble.png"))


print("Testing ScribbleVideoPreprocessor")
pre_v = ScribbleVideoPreprocessor()
out_v = pre_v(video)
export_to_video(out_v.frames, os.path.join(test_dir, "scribble_video.mp4"), fps=24)


