import cv2
import os
from src.detector import PoseDetector
from src.converter import MixamoConverter
from src.exporter import BVHExporter
from src.normalizer import MixamoNormalizer

# 相対path
BASE_DIR   = "."
BONE_JSON  = "data/outputs/bone_lengths.json"
OUTPUT_DIR = "data/outputs"

def process_video(video_path: str, output_path: str):
    detector   = PoseDetector(static_image_mode=False)
    converter  = MixamoConverter()
    normalizer = MixamoNormalizer(BONE_JSON)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] 動画が開けません: {video_path}")
        return

    fps          = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count_str = str(total_frames) if total_frames > 0 else "不明"
    print(f"[動画モード] FPS: {fps:.2f}, 総フレーム数: {frame_count_str}")

    exporter = BVHExporter(frame_rate=int(fps))

    all_frames = []
    frame_idx  = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        landmarks, viz_frame = detector.detect_frame_with_visualization(frame)

        if landmarks:
            normalized = normalizer.normalize(landmarks)
            bone_data  = converter.convert(normalized)
            all_frames.append(bone_data)
        else:
            print(f"[WARN] フレーム {frame_idx}: 骨格検出できず")

        cv2.imshow("Pose Detection", viz_frame)

        if total_frames > 0:
            print(f"\rフレーム処理中: {frame_idx + 1}/{total_frames}", end="")
        else:
            print(f"\rフレーム処理中: {frame_idx + 1}", end="")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()
    detector.close()

    if all_frames:
        exporter.export(all_frames[0], output_path, frames=all_frames)
        print(f"\n[OK] 処理完了: {frame_idx}フレーム → {output_path}")
    else:
        print("\n[ERROR] 有効なフレームがありませんでした")

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    process_video(
        video_path  = "data/videos/sample.mp4",
        output_path = "data/outputs/sample_normalized.bvh"
    )

if __name__ == "__main__":
    main()