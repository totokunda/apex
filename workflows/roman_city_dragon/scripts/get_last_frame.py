import cv2 
import os 

os.makedirs('assets/images', exist_ok=True)
    
path = '/data/apex/workflows/roman_city_dragon/assets/scenes/scene0.mp4'
video = cv2.VideoCapture(path)

num_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
print(f"Number of frames: {num_frames}")

frames = []
while True:
    ret, frame = video.read()
    if not ret:
        break
    frames.append(frame)

cv2.imwrite('assets/images/last_frame.png', frames[-1])