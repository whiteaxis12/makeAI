import json
import numpy as np
from src.detector import PoseLandmark

class MixamoNormalizer:
    """MediaPipe骨格をMixamoボーン長さに正規化"""

    # MediaPipeボーンペア → Mixamoボーン名の対応
    BONE_LENGTH_MAP = {
        "Hips":        ("LEFT_HIP",      "RIGHT_HIP"),
        "Spine":       ("LEFT_HIP",      "LEFT_SHOULDER"),
        "Neck":        ("LEFT_SHOULDER", "NOSE"),
        "LeftArm":     ("LEFT_SHOULDER", "LEFT_ELBOW"),
        "LeftForeArm": ("LEFT_ELBOW",    "LEFT_WRIST"),
        "LeftHand":    ("LEFT_WRIST",    "LEFT_INDEX"),
        "RightArm":    ("RIGHT_SHOULDER","RIGHT_ELBOW"),
        "RightForeArm":("RIGHT_ELBOW",   "RIGHT_WRIST"),
        "RightHand":   ("RIGHT_WRIST",   "RIGHT_INDEX"),
        "LeftUpLeg":   ("LEFT_HIP",      "LEFT_KNEE"),
        "LeftLeg":     ("LEFT_KNEE",     "LEFT_ANKLE"),
        "LeftFoot":    ("LEFT_ANKLE",    "LEFT_HEEL"),
        "RightUpLeg":  ("RIGHT_HIP",     "RIGHT_KNEE"),
        "RightLeg":    ("RIGHT_KNEE",    "RIGHT_ANKLE"),
        "RightFoot":   ("RIGHT_ANKLE",   "RIGHT_HEEL"),
    }

    def __init__(self, bone_lengths_json: str):
        """
        bone_lengths_json: blender_fbx_to_bone_lengths.pyで出力したJSONパス
        """
        with open(bone_lengths_json, "r") as f:
            raw = json.load(f)

        # "mixamorig:"プレフィックスを除去して格納
        self.mixamo_lengths = {}
        for key, val in raw.items():
            clean_key = key.replace("mixamorig:", "")
            self.mixamo_lengths[clean_key] = val["length"]

        print(f"[OK] Mixamoボーン長さを読み込みました: {len(self.mixamo_lengths)}ボーン")

    def _get_pos(self, lm_dict: dict, name: str) -> np.ndarray:
        lm = lm_dict.get(name)
        if lm is None:
            return np.zeros(3)
        return np.array([lm.x, -lm.y, -lm.z])

    def _calc_scale_factor(self, landmarks: list[PoseLandmark]) -> float:
        """
        MediaPipe全身の高さとMixamoの全身の高さから
        グローバルスケールを計算
        """
        lm_dict = {lm.name: lm for lm in landmarks}

        # MediaPipeの全身高さ（頭〜足）
        nose_pos   = self._get_pos(lm_dict, "NOSE")
        ankle_pos  = self._get_pos(lm_dict, "LEFT_ANKLE")
        mp_height  = np.linalg.norm(nose_pos - ankle_pos)

        if mp_height < 1e-6:
            return 1.0

        # Mixamoの全身高さ（主要ボーンの合計）
        mixamo_height = sum([
            self.mixamo_lengths.get("Spine",    0),
            self.mixamo_lengths.get("Spine1",   0),
            self.mixamo_lengths.get("Spine2",   0),
            self.mixamo_lengths.get("Neck",     0),
            self.mixamo_lengths.get("LeftUpLeg",0),
            self.mixamo_lengths.get("LeftLeg",  0),
        ])

        scale = mixamo_height / (mp_height * 100)  # mをcmに変換
        print(f"[OK] スケールファクター: {scale:.4f} (MP高さ:{mp_height:.4f}m, Mixamo高さ:{mixamo_height:.2f}cm)")
        return scale

    def normalize(self, landmarks: list[PoseLandmark]) -> list[PoseLandmark]:
        """
        MediaPipeランドマークをMixamoスケールに正規化
        Returns: スケール済みPoseLandmarkのリスト
        """
        scale = self._calc_scale_factor(landmarks)

        normalized = []
        for lm in landmarks:
            normalized.append(PoseLandmark(
                name=lm.name,
                x=lm.x * scale,
                y=lm.y * scale,
                z=lm.z * scale,
                visibility=lm.visibility
            ))

        return normalized