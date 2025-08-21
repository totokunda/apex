import torch
from src.helpers import WanFunCamera

camera_preprocessor = WanFunCamera()

pose_file = 'assets/wan/poses/pan_left.txt'
H = 480
W = 640
device = torch.device('cpu')
poses = camera_preprocessor(pose_file, H, W, device)

print(poses.shape)
