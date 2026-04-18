import numpy as np
from src.detector import PoseLandmark

class MixamoConverter:
    """MediaPipe骨格座標 → Mixamoボーン回転角に変換"""

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

    # ボーンの位置基準点（中点 or 単点）
    BONE_POSITIONS = {
        "Hips":         ("LEFT_HIP",       "RIGHT_HIP"),
        "Spine":        ("LEFT_HIP",       "RIGHT_HIP"),
        "Spine1":       ("LEFT_HIP",       "RIGHT_HIP"),
        "Spine2":       ("LEFT_SHOULDER",  "RIGHT_SHOULDER"),
        "Neck":         ("LEFT_SHOULDER",  "RIGHT_SHOULDER"),
        "Head":         ("LEFT_SHOULDER",  "RIGHT_SHOULDER"),
        "LeftArm":      ("LEFT_SHOULDER",  None),
        "LeftForeArm":  ("LEFT_ELBOW",     None),
        "LeftHand":     ("LEFT_WRIST",     None),
        "RightArm":     ("RIGHT_SHOULDER", None),
        "RightForeArm": ("RIGHT_ELBOW",    None),
        "RightHand":    ("RIGHT_WRIST",    None),
        "LeftUpLeg":    ("LEFT_HIP",       None),
        "LeftLeg":      ("LEFT_KNEE",      None),
        "LeftFoot":     ("LEFT_ANKLE",     None),
        "LeftToeBase":  ("LEFT_HEEL",      None),
        "RightUpLeg":   ("RIGHT_HIP",      None),
        "RightLeg":     ("RIGHT_KNEE",     None),
        "RightFoot":    ("RIGHT_ANKLE",    None),
        "RightToeBase": ("RIGHT_HEEL",     None),
    }

    BIND_POSE = {
        "Hips":          np.array([1,  0,  0]),
        "Spine":         np.array([0,  1,  0]),
        "Spine1":        np.array([0,  1,  0]),
        "Spine2":        np.array([0,  1,  0]),
        "Neck":          np.array([0,  1,  0]),
        "Head":          np.array([0,  1,  0]),

        "LeftArm":       np.array([1,  0,  0]),
        "LeftForeArm":   np.array([1,  0,  0]),
        "LeftHand":      np.array([1,  0,  0]),

        "RightArm":      np.array([-1, 0,  0]),
        "RightForeArm":  np.array([-1, 0,  0]),
        "RightHand":     np.array([-1, 0,  0]),

        "LeftUpLeg":     np.array([0, -1,  0]),
        "LeftLeg":       np.array([0, -1,  0]),
        "LeftFoot":      np.array([0, -1,  0]),
        "LeftToeBase":   np.array([0,  0,  1]),

        "RightUpLeg":    np.array([0, -1,  0]),
        "RightLeg":      np.array([0, -1,  0]),
        "RightFoot":     np.array([0, -1,  0]),
        "RightToeBase":  np.array([0,  0,  1]),
    }

    CONFIDENCE_THRESHOLD = 0.3

    def __init__(self):
        pass

    def _get_position(self, landmarks: dict, name: str) -> np.ndarray:
        """ランドマーク名から3D座標を取得"""
        lm = landmarks.get(name)
        if lm is None:
            return np.zeros(3)
        return np.array([-lm.x, -lm.y, -lm.z])

    def _get_bone_position(self, lm_dict: dict, bone_name: str) -> np.ndarray:
        """ボーンの位置を取得（中点 or 単点）"""
        p1_name, p2_name = self.BONE_POSITIONS[bone_name]
        p1 = self._get_position(lm_dict, p1_name)
        if p2_name is None:
            return p1
        p2 = self._get_position(lm_dict, p2_name)
        return (p1 + p2) / 2.0

    def _normalize(self, v: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(v)
        if norm < 1e-6:
            return v
        return v / norm

    def _rotation_matrix_from_vectors(self, from_vec: np.ndarray, to_vec: np.ndarray) -> np.ndarray:
        from_vec = self._normalize(from_vec)
        to_vec   = self._normalize(to_vec)
        cos_angle = np.clip(np.dot(from_vec, to_vec), -1.0, 1.0)

        if cos_angle > 0.9999:
            return np.eye(3)
        if cos_angle < -0.9999:
            perp = np.array([1, 0, 0])
            if abs(from_vec[0]) > 0.9:
                perp = np.array([0, 1, 0])
            axis = self._normalize(np.cross(from_vec, perp))
            return self._axis_angle_to_matrix(axis, np.pi)

        axis  = self._normalize(np.cross(from_vec, to_vec))
        angle = np.arccos(cos_angle)
        return self._axis_angle_to_matrix(axis, angle)

    def _axis_angle_to_matrix(self, axis: np.ndarray, angle: float) -> np.ndarray:
        x, y, z = axis
        c = np.cos(angle)
        s = np.sin(angle)
        t = 1 - c
        return np.array([
            [t*x*x + c,   t*x*y - s*z, t*x*z + s*y],
            [t*x*y + s*z, t*y*y + c,   t*y*z - s*x],
            [t*x*z - s*y, t*y*z + s*x, t*z*z + c  ]
        ])

    def _matrix_to_euler_zxy(self, R: np.ndarray) -> np.ndarray:
        sy = np.clip(R[0, 2], -1.0, 1.0)
        if abs(sy) < 0.9999:
            rx = np.arctan2(-R[1, 2], R[2, 2])
            ry = np.arcsin(sy)
            rz = np.arctan2(-R[0, 1], R[0, 0])
        else:
            rx = np.arctan2(R[2, 1], R[1, 1])
            ry = np.pi / 2 * np.sign(sy)
            rz = 0.0
        return np.degrees(np.array([rx, ry, rz]))

    def convert(self, landmarks: list[PoseLandmark]) -> dict:
        lm_dict = {lm.name: lm for lm in landmarks}
        result  = {}

        for bone_name, (parent_name, child_name) in self.BONE_MAPPING.items():
            parent_pos  = self._get_position(lm_dict, parent_name)
            child_pos   = self._get_position(lm_dict, child_name)
            current_dir = self._normalize(child_pos - parent_pos)
            bind_dir    = self.BIND_POSE[bone_name]
            R           = self._rotation_matrix_from_vectors(bind_dir, current_dir)
            rotation    = self._matrix_to_euler_zxy(R)

            # 正しい位置を使用
            bone_pos = self._get_bone_position(lm_dict, bone_name)

            parent_lm  = lm_dict.get(parent_name)
            child_lm   = lm_dict.get(child_name)
            confidence = min(
                parent_lm.visibility if parent_lm else 0.0,
                child_lm.visibility  if child_lm  else 0.0
            )

            result[bone_name] = {
                "position":   bone_pos.tolist(),
                "rotation":   rotation.tolist(),
                "confidence": round(confidence, 3)
            }

            print(f"{bone_name:20s} rot=({rotation[0]:7.2f}, {rotation[1]:7.2f}, {rotation[2]:7.2f}) conf={confidence:.2f}")

        return result