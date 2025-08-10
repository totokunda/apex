from src.preprocess.pose import (
    PosePreprocessor,
    PoseBodyFacePreprocessor,
    PoseBodyFaceVideoPreprocessor,
    PoseBodyPreprocessor,
    PoseBodyVideoPreprocessor,
)
from PIL import Image
from diffusers.utils import export_to_video
import os


# Inputs
image = "assets/image/couple.jpg"
video = "assets/video/conversation.mp4"

# Ensure output directory exists
test_dir = "assets/test"
os.makedirs(test_dir, exist_ok=True)


# Full pose (body, face, hand)
print("Testing PosePreprocessor")
pose_preprocessor = PosePreprocessor()
pose_output = pose_preprocessor(image)

if getattr(pose_output, "detected_map_body", None) is not None:
    Image.fromarray(pose_output.detected_map_body).save(
        os.path.join(test_dir, "pose_body_all.png")
    )
if getattr(pose_output, "detected_map_face", None) is not None:
    Image.fromarray(pose_output.detected_map_face).save(
        os.path.join(test_dir, "pose_face.png")
    )
if getattr(pose_output, "detected_map_bodyface", None) is not None:
    Image.fromarray(pose_output.detected_map_bodyface).save(
        os.path.join(test_dir, "pose_bodyface_all.png")
    )
if getattr(pose_output, "detected_map_handbodyface", None) is not None:
    Image.fromarray(pose_output.detected_map_handbodyface).save(
        os.path.join(test_dir, "pose_handbodyface.png")
    )


# Body + Face (no hands)
print("Testing PoseBodyFacePreprocessor")
pose_bf_preprocessor = PoseBodyFacePreprocessor()
pose_bf_output = pose_bf_preprocessor(image)
Image.fromarray(pose_bf_output.detected_map_bodyface).save(
    os.path.join(test_dir, "pose_bodyface.png")
)


# Body only
print("Testing PoseBodyPreprocessor")
pose_b_preprocessor = PoseBodyPreprocessor()
pose_b_output = pose_b_preprocessor(image)
Image.fromarray(pose_b_output.detected_map_body).save(
    os.path.join(test_dir, "pose_body.png")
)


# Video: Body + Face (no hands)
print("Testing PoseBodyFaceVideoPreprocessor")
pose_bf_video_preprocessor = PoseBodyFaceVideoPreprocessor()
pose_bf_video_output = pose_bf_video_preprocessor(video)
bf_frames = [
    Image.fromarray(frame.detected_map_bodyface) for frame in pose_bf_video_output.frames
]
export_to_video(bf_frames, os.path.join(test_dir, "pose_bodyface_video.mp4"), fps=24)


# Video: Body only
print("Testing PoseBodyVideoPreprocessor")
pose_b_video_preprocessor = PoseBodyVideoPreprocessor()
pose_b_video_output = pose_b_video_preprocessor(video)
b_frames = [
    Image.fromarray(frame.detected_map_body) for frame in pose_b_video_output.frames
]
export_to_video(b_frames, os.path.join(test_dir, "pose_body_video.mp4"), fps=24)


