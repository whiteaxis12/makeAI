# MCAI

※自作AIです。完成まではpythonで作成、スピードを要するならC++に順次移行
- マーカーレスモーションキャプチャーのAIを作成

## 進捗
- 2026/4/17 土台作り、環境構築
- 2026/4/18 画像から頂点を取得→vbhファイル出力、blenderまで持っていく

- 次、オイラー回転修正、boneオフセット修正

## 目次

- [DevelopmentEnviroment](#developmentenviroment)
- [Directory](#Directory)
- [HowTo](#HowTo)


--------





















# DevelopmentEnviroment

- conda 環境python3.11

```cpp
// nvidia-smiにてcuda versionを確認。GPUに合わせたpytorchをinstall
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia

// 0.10.33はpipellineが不安定(solution Error)、0.10.14を明示的にinstall
pip install mediapipe==0.10.14

// リターゲットに必要なFBXを読み込む
pip install pyassimp
```

---

















# Directory

- ざっくりこんな感じ

```

│
├── 📁 data/                    # 入力データ
│   ├── images/                 # 入力画像
│   ├── videos/                 # 入力動画
│   └── outputs/                # 出力結果
│       ├── landmarks/          # 3D頂点データ（JSON/CSV）
│       └── visualizations/     # 可視化画像・動画
│
├── 📁 models/                  # 学習済みモデルの重み
│   └── .gitkeep
│
├── 📁 src/
│   ├── detector.py       # MediaPipeで3D骨格取得
│   ├── converter.py      # 骨格データ → BVH変換
    ├── visualizer.py     # 確認用可視化
    └── exporter.py       # BVH/FBXファイル出力
    ├── normalizer.py     ← ：Mixamoスケールに正規化
    └── fbx_reader.py     ← ：MixamoFBXのボーン長さ読み込み
│
├── 📁 notebooks/               # Jupyter Notebook（実験・検証用）
│   └── 01_prototype.ipynb
│
├── 📁 tests/                   # テストコード
│   └── test_detector.py
│
├── 📁 config/                  # 設定ファイル
│   └── settings.yaml           # パラメータ設定
│
├── main.py                     # エントリーポイント
├── requirements.txt            # pip用ライブラリリスト
├── environment.yml             # conda環境ファイル
└── README.md                   # プロジェクト説明
```

------


















# HowTo

- 作成途中


- MotionCuptureImage
<img src="Date/caputure_Image.png" alt="sampleimage" width="750">