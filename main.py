import mediapipe as mp
import torch
import cv2
import open3d as o3d

print("MediaPipe:", mp.__version__)
print("PyTorch:", torch.__version__)
print("GPU使用可能:", torch.cuda.is_available())
print("OpenCV:", cv2.__version__)
