from src.preprocess.layout.layout import LayoutBboxPreprocessor, LayoutMaskPreprocessor, LayoutTrackPreprocessor
from diffusers.utils import export_to_video
from PIL import Image

mask = "assets/mask/couple_mask.png"
video = "assets/video/conversation.mp4"

preprocessor = LayoutBboxPreprocessor()
bbox = [[0.1, 0.1, 0.2, 0.2], [0.9, 0.9, 0.8, 0.8]]
output = preprocessor(bbox=bbox, frame_size=[256, 256], num_frames=96)

frames = [Image.fromarray(frame) for frame in output.frames]

export_to_video(frames, "assets/test/layout_bbox.mp4", fps=24)

preprocessor = LayoutMaskPreprocessor()
output = preprocessor(mask=mask)

frames = [Image.fromarray(frame) for frame in output.frames]

frames[0].save("assets/test/layout_mask.png")

preprocessor = LayoutTrackPreprocessor()
output = preprocessor(frames=video, mode="salientbboxtrack")

frames = [Image.fromarray(frame) for frame in output.frames]

export_to_video(frames, "assets/test/layout_track.mp4", fps=24)
