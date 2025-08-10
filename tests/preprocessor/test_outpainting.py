from re import I
from src.preprocess.outpainting import OutpaintingPreprocessor, OutpaintingInnerPreprocessor, OutpaintingVideoPreprocessor, OutpaintingInnerVideoPreprocessor
from PIL import Image
from diffusers.utils import export_to_video

image = "assets/image/couple.jpg"
mask = "assets/mask/couple_mask.png"
video = "assets/video/conversation.mp4"

preprocessor = OutpaintingPreprocessor()
output = preprocessor(image=image, expand_ratio=0.3)

output.mask.save("assets/test/outpainting.png")

output = preprocessor(image=image, expand_ratio=0.3, mask=mask)

output.mask.save("assets/test/outpainting_mask.png")

preprocessor = OutpaintingInnerPreprocessor()
output = preprocessor(image=image, expand_ratio=0.3)

output.mask.save("assets/test/outpainting_inner.png")

preprocessor = OutpaintingVideoPreprocessor()
output = preprocessor(frames=video, expand_ratio=0.3)

export_to_video(output.masks, "assets/test/outpainting_video.mp4")

preprocessor = OutpaintingInnerVideoPreprocessor()
output = preprocessor(frames=video, expand_ratio=0.3)
export_to_video(output.masks, "assets/test/outpainting_inner_video.mp4")

preprocessor = OutpaintingVideoPreprocessor()
output = preprocessor(frames=video, expand_ratio=0.3, mask=mask)
export_to_video(output.masks, "assets/test/outpainting_video_mask.mp4")





