from src.preprocess.frameref import FrameRefExpandPreprocessor, FrameRefExtractPreprocessor
from diffusers.utils import export_to_video
from PIL import Image

video = "assets/video/conversation.mp4"
video2 = "assets/video/jail.mp4"

preprocessor = FrameRefExtractPreprocessor()
output = preprocessor(video)
output.frames = [Image.fromarray(frame) for frame in output.frames]
export_to_video(output.frames, "assets/test/frameref.mp4", fps=24)

preprocessor = FrameRefExpandPreprocessor()
output = preprocessor(
    frames=video,
    image_2="/Users/tosinkuye/apex/assets/image/dog.png",
    mode="firstlastframe",
    expand_num=96,
    return_mask=True,
    resize=True,
)
output.frames = [Image.fromarray(frame) for frame in output.frames]
export_to_video(output.frames, "assets/test/frameref_expand.mp4", fps=24)