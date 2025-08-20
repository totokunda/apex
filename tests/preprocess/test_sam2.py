from src.preprocess.sam2 import SAM2Preprocessor, SAM2VideoPreprocessor
from diffusers.utils import export_to_video
from PIL import Image
import os


image = "assets/image/couple.jpg"
video = "assets/video/conversation.mp4"
mask = "assets/mask/couple_mask.png"

test_dir = "assets/test"
os.makedirs(test_dir, exist_ok=True)


print("Testing SAM2Preprocessor with input_box")
sam2 = SAM2Preprocessor(task_type="input_box", return_mask=True)
output = sam2(image=image, input_box=[0.1, 0.1, 0.8, 0.8])
if output.masks is not None:
    Image.fromarray((output.masks * 255).astype("uint8")).save(
        os.path.join(test_dir, "sam2_input_box_mask.png")
    )


print("Testing SAM2Preprocessor with mask input (mask)")
sam2 = SAM2Preprocessor(task_type="mask", return_mask=True)
output = sam2(image=image, mask=mask)
if output.masks is not None:
    Image.fromarray((output.masks * 255).astype("uint8")).save(
        os.path.join(test_dir, "sam2_mask_input.png")
    )


print("Testing SAM2Preprocessor with mask_point")
sam2 = SAM2Preprocessor(task_type="mask_point", return_mask=True)
output = sam2(image=image, mask=mask)
if output.masks is not None:
    Image.fromarray((output.masks * 255).astype("uint8")).save(
        os.path.join(test_dir, "sam2_mask_point.png")
    )


print("Testing SAM2Preprocessor with mask_box")
sam2 = SAM2Preprocessor(task_type="mask_box", return_mask=True)
output = sam2(image=image, mask=mask)
if output.masks is not None:
    Image.fromarray((output.masks * 255).astype("uint8")).save(
        os.path.join(test_dir, "sam2_mask_box.png")
    )


print("Testing SAM2VideoPreprocessor with input_box")
sam2_video = SAM2VideoPreprocessor(task_type="input_box")
video_output = sam2_video(video=video, input_box=[0.1, 0.1, 0.9, 0.9])
# Convert annotations to per-frame masks and export
from src.preprocess.inpainting.inpainting import single_rle_to_mask

frame_masks = []
for idx in sorted(video_output.annotations.keys()):
    frame_info = video_output.annotations[idx]
    # Use first object mask if exists
    if len(frame_info) == 0:
        continue
    first_obj = next(iter(frame_info.values()))
    mask = single_rle_to_mask(first_obj["mask"]).astype("uint8") * 255
    frame_masks.append(Image.fromarray(mask))

if frame_masks:
    export_to_video(frame_masks, os.path.join(test_dir, "sam2_video_input_box.mp4"), fps=24)


