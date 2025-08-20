from src.preprocess.composition.composition import CompositionPreprocessor
from diffusers.utils import export_to_video

preprocessor = CompositionPreprocessor()
video_1 = "assets/video/conversation.mp4"
video_2 = "assets/video/jail.mp4"
mask_1 = "assets/mask/conversation_mask.mp4"
mask_2 = "assets/mask/jail_mask.mp4"

output = preprocessor(
    process_type_1="control",
    process_type_2="control",
    video_1=video_1,
    video_2=video_2,
    mask_1=mask_1,
    mask_2=mask_2,
)

video = output.video
mask = output.mask

# save video and mask
export_to_video(video, "assets/video/conversation_jail.mp4", fps=24)
export_to_video(mask, "assets/mask/conversation_jail_mask.mp4", fps=24)