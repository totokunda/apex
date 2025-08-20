from src.preprocess.composition.composition import SwapAnythingPreprocessor
from diffusers.utils import export_to_video
import os 
from tqdm import tqdm

preprocessor = SwapAnythingPreprocessor()
test_dir = "assets/test"
os.makedirs(test_dir, exist_ok=True)

video = "assets/video/conversation.mp4"
images = ["assets/image/couple.jpg"]
mask = "assets/mask/couple_mask.png"
bbox = [0.2, 0.2, 0.8, 0.8]
caption = "Two people are talking"
label = "person"

RUN_TESTS = [
    #"salientbboxtrack,salientbboxtrack",
    "maskbboxtrack,maskbboxtrack",
    "masktrack,mask",
    "bbox,bbox",
    "maskpointtrack,maskpointtrack",
    "bboxtrack,bboxtrack",
    "label,label",
    "caption,caption",
]
for test in tqdm(RUN_TESTS):

    output = preprocessor(
        frames=video,
        images=images,
        mask=mask,
        mode=test,
        bbox=bbox,
        caption=caption,
        label=label,
        return_mask=True,
    )

    export_to_video(output.video, os.path.join(test_dir, f"conversation_swap_anything_{test.replace(',', '_')}.mp4"), fps=24)
    export_to_video(output.mask, os.path.join(test_dir, f"conversation_swap_anything_mask_{test.replace(',', '_')}.mp4"), fps=24)