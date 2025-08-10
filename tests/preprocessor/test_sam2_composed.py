from src.preprocess.sam2.sam2 import (
    SAM2SalientVideoPreprocessor,
    SAM2GDINOVideoPreprocessor,
)
from src.preprocess.inpainting.inpainting import single_rle_to_mask
from diffusers.utils import export_to_video
from PIL import Image
import os


video = "assets/video/conversation.mp4"
test_dir = "assets/test"
os.makedirs(test_dir, exist_ok=True)


print("Testing SAM2SalientVideoPreprocessor")
sam2_salient = SAM2SalientVideoPreprocessor()
salient_out = sam2_salient(video)

salient_masks = []
for idx in sorted(salient_out.annotations.keys()):
    frame_info = salient_out.annotations[idx]
    if len(frame_info) == 0:
        continue
    first_obj = next(iter(frame_info.values()))
    mask = single_rle_to_mask(first_obj["mask"]).astype("uint8") * 255
    salient_masks.append(Image.fromarray(mask))

if salient_masks:
    export_to_video(salient_masks, os.path.join(test_dir, "sam2_salient_video.mp4"), fps=24)


print("Testing SAM2GDINOVideoPreprocessor (caption)")
sam2_gdino = SAM2GDINOVideoPreprocessor()
gdino_out = sam2_gdino(video, caption="a person")

gdino_masks = []
for idx in sorted(gdino_out.annotations.keys()):
    frame_info = gdino_out.annotations[idx]
    if len(frame_info) == 0:
        continue
    first_obj = next(iter(frame_info.values()))
    mask = single_rle_to_mask(first_obj["mask"]).astype("uint8") * 255
    gdino_masks.append(Image.fromarray(mask))

if gdino_masks:
    export_to_video(gdino_masks, os.path.join(test_dir, "sam2_gdino_video.mp4"), fps=24)


