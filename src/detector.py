import cv2
import mediapipe as mp
import numpy as np
from dataclasses import dataclass
from typing import Optional

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

    def __init__(self,
                 static_image_mode: bool = False,
                 model_complexity: int = 2,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):

        self.mp_pose   = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

    def detect_frame(self, frame) -> Optional[list[PoseLandmark]]:
        """
        フレーム（numpy配列）から骨格検出
        動画・カメラ両方で使用
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.pose.process(rgb)
        rgb.flags.writeable = True

        if not results.pose_world_landmarks:
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
        return landmarks

    def detect_frame_with_visualization(self, frame):
        """
        フレームから骨格検出＋描画
        Returns: (landmarks, 描画済みフレーム)
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.pose.process(rgb)
        rgb.flags.writeable = True

        if not results.pose_world_landmarks:
            return None, frame

        self.mp_drawing.draw_landmarks(
            frame,
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
        return landmarks, frame

    def detect(self, image_path: str) -> Optional[list[PoseLandmark]]:
        """静止画から骨格検出"""
        image = cv2.imread(image_path)
        if image is None:
            print(f"[ERROR] 画像が読み込めません: {image_path}")
            return None
        return self.detect_frame(image)

    def detect_with_visualization(self, image_path: str):
        """静止画から骨格検出＋描画"""
        image = cv2.imread(image_path)
        if image is None:
            print(f"[ERROR] 画像が読み込めません: {image_path}")
            return None, None
        return self.detect_frame_with_visualization(image)

    def close(self):
        self.pose.close()