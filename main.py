import cv2
import os
from src.detector import PoseDetector
from src.converter import MixamoConverter
from src.exporter import BVHExporter

def main():
    detector  = PoseDetector()
    converter = MixamoConverter()
    exporter  = BVHExporter(frame_rate=30)

    image_path = r"C:\Users\naoki\Downloads\testPose.jpg"
    output_path = r"C:\Users\naoki\Downloads\sample.bvh"
        
        
    # 出力フォルダ作成
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"画像を処理中: {image_path}\n")

    # Step1: 検出
    landmarks, result_image = detector.detect_with_visualization(image_path)
    if landmarks is None:
        print("骨格が検出できませんでした")
        return

    # Step2: 変換
    print("\n=== Mixamoボーン回転角 ===")
    bone_data = converter.convert(landmarks)

    # Step3: BVH出力
    print("\n=== BVH出力 ===")
    exporter.export(bone_data, output_path)

    # 画像表示
    cv2.imshow("Pose Detection", result_image)
    print("\n[INFO] 何かキーを押すと終了します")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    detector.close()

if __name__ == "__main__":
    main()