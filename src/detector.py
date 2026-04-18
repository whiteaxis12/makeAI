import cv2
import mediapipe as mp
import numpy as np
from dataclasses import dataclass
from typing import Optional
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

@dataclass
class PoseLandmark:
    """1つの骨格点データ"""
    name: str
    x: float
    y: float
    z: float
    visibility: float

class PoseDetector:
    """MediaPipeを使った人体3D骨格検出"""

    LANDMARK_NAMES = [
        "NOSE", "LEFT_EYE_INNER", "LEFT_EYE", "LEFT_EYE_OUTER",
        "RIGHT_EYE_INNER", "RIGHT_EYE", "RIGHT_EYE_OUTER",
        "LEFT_EAR", "RIGHT_EAR",
        "MOUTH_LEFT", "MOUTH_RIGHT",
        "LEFT_SHOULDER", "RIGHT_SHOULDER",
        "LEFT_ELBOW", "RIGHT_ELBOW",
        "LEFT_WRIST", "RIGHT_WRIST",
        "LEFT_PINKY", "RIGHT_PINKY",
        "LEFT_INDEX", "RIGHT_INDEX",
        "LEFT_THUMB", "RIGHT_THUMB",
        "LEFT_HIP", "RIGHT_HIP",
        "LEFT_KNEE", "RIGHT_KNEE",
        "LEFT_ANKLE", "RIGHT_ANKLE",
        "LEFT_HEEL", "RIGHT_HEEL",
        "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX"
    ]

    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        # 新しいAPIの書き方
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            enable_segmentation=False
        )

    def detect(self, image_path: str) -> Optional[list[PoseLandmark]]:
        image = cv2.imread(image_path)
        if image is None:
            print(f"[ERROR] 画像が読み込めません: {image_path}")
            return None

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.pose.process(rgb)
        rgb.flags.writeable = True

        if not results.pose_world_landmarks:
            print("[ERROR] 骨格が検出できませんでした")
            return None

        landmarks = []
        for i, lm in enumerate(results.pose_world_landmarks.landmark):
            landmarks.append(PoseLandmark(
                name=self.LANDMARK_NAMES[i],
                x=lm.x,
                y=lm.y,
                z=lm.z,
                visibility=lm.visibility
            ))

        print(f"[OK] {len(landmarks)}点の骨格を検出しました")
        return landmarks

    def detect_with_visualization(self, image_path: str):
        image = cv2.imread(image_path)
        if image is None:
            print(f"[ERROR] 画像が読み込めません: {image_path}")
            return None, None

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.pose.process(rgb)
        rgb.flags.writeable = True

        if not results.pose_world_landmarks:
            print("[ERROR] 骨格が検出できませんでした")
            return None, image

        self.mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            self.mp_pose.POSE_CONNECTIONS
        )

        landmarks = []
        for i, lm in enumerate(results.pose_world_landmarks.landmark):
            landmarks.append(PoseLandmark(
                name=self.LANDMARK_NAMES[i],
                x=lm.x,
                y=lm.y,
                z=lm.z,
                visibility=lm.visibility
            ))

        return landmarks, image

    def close(self):
        self.pose.close()