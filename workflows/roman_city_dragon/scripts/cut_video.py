import cv2
import PIL
from diffusers.utils import export_to_video

video_path = '/data/apex/workflows/roman_city_dragon/assets/flux_kontext/dragon_flux_kontext_continuation_1.mp4'
output_path = '/data/apex/workflows/roman_city_dragon/assets/flux_kontext/dragon_flux_kontext_continuation_1_cut.mp4'
video = cv2.VideoCapture(video_path)

fps = video.get(cv2.CAP_PROP_FPS)
#
start_frame = 0 
end_frame = int(2.7 * fps)


frames = []
while True:
    ret, frame = video.read()
    if not ret:
        break
    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


frames = frames[start_frame:end_frame]

frames = [PIL.Image.fromarray(frame) for frame in frames]

export_to_video(frames, output_path, fps=fps, quality=9)
