from src.preprocess.flow import FlowPreprocessor
from diffusers.utils import export_to_video

video = "assets/video/conversation.mp4"

print("Testing FlowPreprocessor")
preprocessor = FlowPreprocessor()

output = preprocessor(video)

export_to_video(output.flow_vis, "assets/test/flow.mp4", fps=10)