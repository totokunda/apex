from src.preprocess.composition.composition import AnimateAnythingPreprocessor
from diffusers.utils import export_to_video
import os 

preprocessor = AnimateAnythingPreprocessor()
test_dir = "assets/test"
os.makedirs(test_dir, exist_ok=True)

video = "assets/video/conversation.mp4"

output = preprocessor(
    frames=video,
    ref_mode="salient",
    return_mask=True,
    mask_cfg=None
)

export_to_video(output.frames, os.path.join(test_dir, "conversation_animate_anything.mp4"), fps=24)