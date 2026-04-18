import cv2
import os
from src.detector import PoseDetector
from src.converter import MixamoConverter
from src.exporter import BVHExporter

def process_image(image_path: str, output_path: str):
    """静止画処理"""
    detector  = PoseDetector(static_image_mode=True)
    converter = MixamoConverter()
    exporter  = BVHExporter(frame_rate=30)

    print(f"[画像モード] {image_path}")

    landmarks, result_image = detector.detect_with_visualization(image_path)
    if landmarks is None:
        print("[ERROR] 骨格が検出できませんでした")
        return

    bone_data = converter.convert(landmarks)
    exporter.export(bone_data, output_path)

    cv2.imshow("Pose Detection", result_image)
    print("[INFO] 何かキーを押すと終了します")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    detector.close()


def process_video(video_path: str, output_path: str):
    """動画処理"""
    detector  = PoseDetector(static_image_mode=False)
    converter = MixamoConverter()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] 動画が開けません: {video_path}")
        return

    fps         = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    exporter    = BVHExporter(frame_rate=int(fps))

    print(f"[動画モード] {video_path}")
    print(f"FPS: {fps}, 総フレーム数: {total_frames}")

    all_frames = []
    frame_idx  = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        landmarks, viz_frame = detector.detect_frame_with_visualization(frame)

        if landmarks:
            bone_data = converter.convert(landmarks)
            all_frames.append(bone_data)
        else:
            print(f"[WARN] フレーム {frame_idx}: 骨格検出できず")

        # プレビュー表示
        cv2.imshow("Pose Detection", viz_frame)
        print(f"\rフレーム処理中: {frame_idx + 1}/{total_frames}", end="")

        # qキーで中断
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n[INFO] 中断しました")
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
    # ===== ここを切り替えて使う =====
    MODE = "video"  # "image" or "video"
    # ================================

    BASE_DIR   = r"C:\Users\naoki\MyProject\SourceCode\repos_vscode\MCAI"
    OUTPUT_DIR = os.path.join(BASE_DIR, "data", "outputs")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if MODE == "image":
        process_image(
            image_path  = os.path.join(BASE_DIR, "data", "images", "sample.jpg"),
            output_path = os.path.join(OUTPUT_DIR, "sample.bvh")
        )
    elif MODE == "video":
        process_video(
            video_path  = os.path.join(BASE_DIR, "data", "videos", "sample.mp4"),
            output_path = os.path.join(OUTPUT_DIR, "sample_video.bvh")
        )

if __name__ == "__main__":
    main()