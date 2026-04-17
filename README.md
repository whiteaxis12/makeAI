# MCAI

※自作AIです。完成まではpythonで作成、スピードを要するならC++に順次移行
- マーカーレスモーションキャプチャーのAIを作成

## 目次

- [DevelopmentEnviroment](#developmentenviroment)


--------





















# DevelopmentEnviroment

- conda 環境python3.11

```cpp
// nvidia-smiにてcuda versionを確認。GPUに合わせたpytorchをinstall
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia

pip install mediapipe opencv-python open3d
```
[def]: #目次