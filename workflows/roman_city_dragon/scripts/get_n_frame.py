import cv2 
import os 

os.makedirs('assets/images', exist_ok=True)
    
path = '/data/apex/workflows/roman_city_dragon/assets/scenes/scene0.mp4'
video = cv2.VideoCapture(path)
frame_number = 49
num_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
frame_number = num_frames - 1 if frame_number is None else frame_number
video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

print(f"Number of frames: {num_frames}")
ret, frame = video.read()
if not ret:
    raise ValueError("Failed to read frame")
cv2.imwrite(f'assets/images/frame_{frame_number}.png', frame)
video.release()