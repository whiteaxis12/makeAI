import cv2
import numpy as np
from src.detector import PoseDetector
from src.converter import MixamoConverter
from src.normalizer import MixamoNormalizer

BONE_JSON  = "data/outputs/bone_lengths.json"

# ボーンの接続関係（描画用）
BONE_CONNECTIONS = [
    ("LEFT_HIP",      "RIGHT_HIP"),
    ("LEFT_HIP",      "LEFT_SHOULDER"),
    ("RIGHT_HIP",     "RIGHT_SHOULDER"),
    ("LEFT_SHOULDER", "RIGHT_SHOULDER"),
    ("LEFT_SHOULDER", "LEFT_ELBOW"),
    ("LEFT_ELBOW",    "LEFT_WRIST"),
    ("RIGHT_SHOULDER","RIGHT_ELBOW"),
    ("RIGHT_ELBOW",   "RIGHT_WRIST"),
    ("LEFT_HIP",      "LEFT_KNEE"),
    ("LEFT_KNEE",     "LEFT_ANKLE"),
    ("RIGHT_HIP",     "RIGHT_KNEE"),
    ("RIGHT_KNEE",    "RIGHT_ANKLE"),
    ("LEFT_SHOULDER", "NOSE"),
    ("NOSE",          "LEFT_EAR"),
]

def draw_landmarks(image, landmarks, color=(0, 255, 0)):
    """MediaPipeのランドマークを2D画像に描画"""
    h, w = image.shape[:2]
    lm_dict = {lm.name: lm for lm in landmarks}

    # 点を描画
    for lm in landmarks:
        # pose_landmarksの正規化座標を使用
        px = int(lm.x * w)  
        py = int(lm.y * h)
        cv2.circle(image, (px, py), 5, color, -1)
        cv2.putText(image, lm.name[:8], (px, py-5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

    return image

def main():
    detector   = PoseDetector(static_image_mode=True)
    normalizer = MixamoNormalizer(BONE_JSON)
    converter  = MixamoConverter()

    image_path = "data/image/sample.jpg"
    image = cv2.imread(image_path)

    # MediaPipe検出（2D座標も取得）
    import mediapipe as mp
    mp_pose    = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

    with mp_pose.Pose(static_image_mode=True, model_complexity=2) as pose:
        rgb     = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        if results.pose_landmarks:
            # 2D描画（赤）
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=4)
            )

        if results.pose_world_landmarks:
            # world_landmarksをログ出力
            print("\n=== MediaPipe 3D座標 (world) ===")
            for i, lm in enumerate(results.pose_world_landmarks.landmark):
                name = detector.LANDMARK_NAMES[i]
                print(f"{name:25s} x={lm.x:7.3f} y={lm.y:7.3f} z={lm.z:7.3f} vis={lm.visibility:.2f}")

    # converter.pyの回転計算結果をログ出力
    landmarks = detector.detect(image_path)
    if landmarks:
        normalized = normalizer.normalize(landmarks)

        print("\n=== 正規化後の座標 ===")
        for lm in normalized:
            print(f"{lm.name:25s} x={lm.x:7.3f} y={lm.y:7.3f} z={lm.z:7.3f}")

        print("\n=== converter.py 回転計算結果 ===")
        bone_data = converter.convert(normalized)
        for bone_name, data in bone_data.items():
            pos = data['position']
            rot = data['rotation']
            print(f"{bone_name:20s} pos=({pos[0]:6.3f},{pos[1]:6.3f},{pos[2]:6.3f}) rot=({rot[0]:7.2f},{rot[1]:7.2f},{rot[2]:7.2f})")

    cv2.imshow("MediaPipe Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()