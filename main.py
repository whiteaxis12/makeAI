import cv2
from src.detector import PoseDetector

def main():
    # 検出器を初期化
    detector = PoseDetector()

    # 画像パス（ここを変更してください）
    image_path = r"C:\Users\naoki\Downloads\testPose.jpg"

    print(f"画像を処理中: {image_path}")

    # 検出 + 可視化
    landmarks, result_image = detector.detect_with_visualization(image_path)

    if landmarks is None:
        print("骨格が検出できませんでした")
        return

    # 結果を表示
    print("\n=== 検出された骨格点 ===")
    for lm in landmarks:
        print(f"{lm.name:25s} x={lm.x:7.3f}, y={lm.y:7.3f}, z={lm.z:7.3f}, visibility={lm.visibility:.2f}")

    # 画像を表示
    cv2.imshow("Pose Detection", result_image)
    print("\n[INFO] 何かキーを押すと終了します")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # リソース解放
    detector.close()

if __name__ == "__main__":
    main()