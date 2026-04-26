import cv2
import os
from src.detector import PoseDetector
from src.converter import MixamoConverter
from src.exporter import BVHExporter
from src.fbx_reader import FBXReader

def test_fbx_reader():

    FrameRate = 24
    deltaTime = 1 / FrameRate
    SetFrameRate = 2
    CurrentDeltatime = 0

    for i in range(100):
        CurrentDeltatime += deltaTime * SetFrameRate 
        print(i, CurrentDeltatime)

# mainの最初に追加
if __name__ == "__main__":
    test_fbx_reader()