from src.preprocess.composition.composition import ExpandAnythingPreprocessor
from diffusers.utils import export_to_video
import os 

preprocessor = ExpandAnythingPreprocessor()
test_dir = "assets/test"
os.makedirs(test_dir, exist_ok=True)

image_1 = "assets/image/couple.jpg"
image_2 = "assets/image/dog.png"

output = preprocessor(
    images=[image_1, image_2],
    mode="firstframe,salient",
    return_mask=True,
    mask_cfg=None,
    expand_ratio=0.3,
    expand_num=96
)

export_to_video(output.frames, os.path.join(test_dir, "expand_anything.mp4"), fps=24)
export_to_video(output.masks, os.path.join(test_dir, "expand_anything_mask.mp4"), fps=24)
export_to_video(output.images, os.path.join(test_dir, "expand_anything_image.mp4"), fps=24)