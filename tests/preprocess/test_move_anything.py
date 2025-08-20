from src.preprocess.composition.composition import MoveAnythingPreprocessor
from diffusers.utils import export_to_video
import os 

preprocessor = MoveAnythingPreprocessor()
test_dir = "assets/test"
os.makedirs(test_dir, exist_ok=True)

image = "assets/image/couple.jpg"
bbox = [[0.2, 0.2, 0.8, 0.8], [0.4, 0.4, 0.6, 0.6]]

output = preprocessor(
    image=image,
    start_bbox=bbox[0],
    end_bbox=bbox[1],
    expand_num=96,
    label="person"
)

export_to_video(output.frames, os.path.join(test_dir, "move_anything.mp4"), fps=24)
export_to_video(output.masks, os.path.join(test_dir, "move_anything_mask.mp4"), fps=24)