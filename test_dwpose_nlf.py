from src.preprocess.dwpose_nlf import DwposeNlfDetector
from src.preprocess.dwpose import DwposeDetector
from src.types import InputImage
from diffusers.utils import export_to_video
detector = DwposeNlfDetector.from_pretrained()
path = "/home/tosin_coverquick_co/apex/dance-masked.mp4"
frames = detector._load_video(path, fps=16)
outframes = [frame for frame in detector.process_video(frames)]
export_to_video(outframes, "rendered_dance_dwpose_nlf.mp4", fps=16)