import numpy as np
from src.detector import PoseLandmark

class MixamoConverter:
    """MediaPipe骨格座標 → Mixamoボーン回転角に変換"""

    # MediaPipe → Mixamoのボーン対応表
    BONE_MAPPING = {
        "Hips":          ("LEFT_HIP",       "RIGHT_HIP"),
        "Spine":         ("LEFT_HIP",       "LEFT_SHOULDER"),
        "Spine1":        ("LEFT_HIP",       "LEFT_SHOULDER"),
        "Spine2":        ("LEFT_SHOULDER",  "RIGHT_SHOULDER"),
        "Neck":          ("LEFT_SHOULDER",  "NOSE"),
        "Head":          ("NOSE",           "LEFT_EAR"),

        "LeftArm":       ("LEFT_SHOULDER",  "LEFT_ELBOW"),
        "LeftForeArm":   ("LEFT_ELBOW",     "LEFT_WRIST"),
        "LeftHand":      ("LEFT_WRIST",     "LEFT_INDEX"),

        "RightArm":      ("RIGHT_SHOULDER", "RIGHT_ELBOW"),
        "RightForeArm":  ("RIGHT_ELBOW",    "RIGHT_WRIST"),
        "RightHand":     ("RIGHT_WRIST",    "RIGHT_INDEX"),

        "LeftUpLeg":     ("LEFT_HIP",       "LEFT_KNEE"),
        "LeftLeg":       ("LEFT_KNEE",      "LEFT_ANKLE"),
        "LeftFoot":      ("LEFT_ANKLE",     "LEFT_HEEL"),
        "LeftToeBase":   ("LEFT_HEEL",      "LEFT_FOOT_INDEX"),

        "RightUpLeg":    ("RIGHT_HIP",      "RIGHT_KNEE"),
        "RightLeg":      ("RIGHT_KNEE",     "RIGHT_ANKLE"),
        "RightFoot":     ("RIGHT_ANKLE",    "RIGHT_HEEL"),
        "RightToeBase":  ("RIGHT_HEEL",     "RIGHT_FOOT_INDEX"),
    }

    def __init__(self):
        pass

    def _get_position(self, landmarks: dict, name: str) -> np.ndarray:
        """ランドマーク名から3D座標を取得"""
        lm = landmarks.get(name)
        if lm is None:
            return np.zeros(3)
        return np.array([lm.x, lm.y, lm.z])

    def _calc_rotation(self, parent_pos: np.ndarray, child_pos: np.ndarray) -> np.ndarray:
        """
        2点間のベクトルからオイラー角（度）を計算
        Returns: [rx, ry, rz] in degrees
        """
        direction = child_pos - parent_pos
        length = np.linalg.norm(direction)

        if length < 1e-6:
            return np.zeros(3)

        direction = direction / length

        # オイラー角に変換
        rx = np.degrees(np.arctan2(direction[1], direction[2]))
        ry = np.degrees(np.arctan2(direction[0], direction[2]))
        rz = np.degrees(np.arctan2(direction[0], direction[1]))

        return np.array([rx, ry, rz])

    def convert(self, landmarks: list[PoseLandmark]) -> dict:
        """
        MediaPipeのランドマークリストをMixamoボーン回転角に変換
        Returns: {ボーン名: {"position": [...], "rotation": [rx, ry, rz]}}
        """
        # リストを辞書に変換（名前でアクセスしやすくする）
        lm_dict = {lm.name: lm for lm in landmarks}

        result = {}

        for bone_name, (parent_name, child_name) in self.BONE_MAPPING.items():
            parent_pos = self._get_position(lm_dict, parent_name)
            child_pos  = self._get_position(lm_dict, child_name)

            rotation = self._calc_rotation(parent_pos, child_pos)

            # visibilityが低い点は信頼度を下げる
            parent_lm = lm_dict.get(parent_name)
            child_lm  = lm_dict.get(child_name)
            confidence = min(
                parent_lm.visibility if parent_lm else 0.0,
                child_lm.visibility  if child_lm  else 0.0
            )

            result[bone_name] = {
                "position":   parent_pos.tolist(),
                "rotation":   rotation.tolist(),
                "confidence": round(confidence, 3)
            }

            print(f"{bone_name:20s} rot=({rotation[0]:7.2f}, {rotation[1]:7.2f}, {rotation[2]:7.2f}) conf={confidence:.2f}")

        return result