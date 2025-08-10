from src.preprocess.salient import SalientPreprocessor, SalientVideoPreprocessor
from diffusers.utils import export_to_video
from PIL import Image
import os


image = "assets/image/dog.png"
video = "assets/video/conversation.mp4"

test_dir = "assets/test"
os.makedirs(test_dir, exist_ok=True)


print("Testing SalientPreprocessor (mask only)")
preprocessor = SalientPreprocessor()
output = preprocessor(image)
Image.fromarray(output.mask).save(os.path.join(test_dir, "salient_mask.png"))


print("Testing SalientPreprocessor (return image)")
preprocessor = SalientPreprocessor()
output = preprocessor(image)
print(output.image.size, output.mask.size)
Image.fromarray(output.image).save(os.path.join(test_dir, "salient_image.png"))
Image.fromarray(output.mask).save(os.path.join(test_dir, "salient_mask_image.png"))


print("Testing SalientVideoPreprocessor (mask frames)")
video_preprocessor = SalientVideoPreprocessor()
video_outputs = video_preprocessor(video)
frames = [Image.fromarray(frame.mask) for frame in video_outputs.frames]
export_to_video(frames, os.path.join(test_dir, "salient_video_mask.mp4"), fps=24)


