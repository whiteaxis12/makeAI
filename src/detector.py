import cv2
import mediapipe as mp
import numpy as np
from dataclasses import dataclass
from typing import Optional

@dataclass
class PoseLandmark:
    """1つの骨格点データ"""
    name: str
    x: float  # 横位置 (0.0〜1.0)
    y: float  # 縦位置 (0.0〜1.0)
    z: float  # 深度
    visibility: float  # 検出信頼度 (0.0〜1.0)

class PoseDetector:
    """MediaPipeを使った人体3D骨格検出"""

    # MediaPipeの33点の名前リスト
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
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,  # 静止画モード
            model_complexity=2,      # 精度最高（0/1/2）
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

    def detect(self, image_path: str) -> Optional[list[PoseLandmark]]:
        """
        画像から3D骨格点を検出する
        Returns: PoseLandmarkのリスト or None（検出失敗時）
        """
        # 画像読み込み
        image = cv2.imread(image_path)
        if image is None:
            print(f"[ERROR] 画像が読み込めません: {image_path}")
            return None

        # BGR → RGB変換
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 骨格検出
        results = self.pose.process(rgb)

        if not results.pose_world_landmarks:
            print("[ERROR] 骨格が検出できませんでした")
            return None

        # 結果をPoseLandmarkに変換
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
        """
        検出結果を画像に描画して返す
        Returns: (landmarks, 描画済み画像)
        """
        image = cv2.imread(image_path)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)

        if not results.pose_world_landmarks:
            return None, image

        # 骨格を画像に描画
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
        """リソース解放"""
        self.pose.close()