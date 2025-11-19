from src.preprocess.pose2d import Pose2dDetector
from PIL import Image

pose2d = Pose2dDetector.from_pretrained()
video = "/home/tosin_coverquick_co/apex/animated_character.mp4"
pose = pose2d(video, mode="pose")
for frame in pose:
    print(frame.size)