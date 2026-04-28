import cv2
import os
from src.detector import PoseDetector
from src.converter import MixamoConverter
from src.exporter import BVHExporter
from src.fbx_reader import FBXReader

def test_fbx_reader():

    fps = 24
    deltaTime = 1 / fps
    NewFps = 2
    PhaseFrameRate = 0
    time = 0

    for i in range(24):
        time += deltaTime
        PhaseFrameRate += deltaTime * NewFps 
        print(f"{i=}, {time=}, {PhaseFrameRate=}")

# mainの最初に追加
if __name__ == "__main__":
    test_fbx_reader()