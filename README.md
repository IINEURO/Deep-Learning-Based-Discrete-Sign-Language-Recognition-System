# Deep Learning Based Discrete Sign Language Recognition System V1.0(基于深度学习的离散手语识别系统）

基于 **CSL+PyTorch + MediaPipe + Gradio** 的离散手语识别基线工程，支持：

- 图片识别
- 视频按时间间隔切片识别（输出时间段结果表）
- 实时摄像头识别（持续输出字符与时间片段）

## 1. 项目特点

- `src` 结构化代码组织（`src/slr_baseline`）
- 训练、离线推理、实时演示、可视化脚本分离
- 支持 `Hands + Pose` 关键点（55点，4维）
- Gradio 可视化界面，适合演示

## 2. 目录结构

```text
Sign-Language-Trans/
├── gradio_app.py
├── train.py
├── infer.py
├── realtime_demo.py
├── scripts/
│   ├── scan_csl.py
│   ├── prepare_meta.py
│   ├── extract_keypoints.py
│   └── visualize_samples.py
├── src/slr_baseline/
├── docs/
├── assets/
├── requirements.txt
└── pyproject.toml
```

详细结构说明见：`docs/PROJECT_STRUCTURE.md`。

## 3. 环境要求

- Linux（推荐 Ubuntu 22.04）
- Python 3.10
- 依赖见 `requirements.txt`

安装：

```bash
pip install -r requirements.txt
```

可选开发安装：

```bash
pip install -e .
```

## 4. 快速开始

### 4.1 启动 UI

```bash
python gradio_app.py
```

默认地址：`http://127.0.0.1:7860`

### 4.2 数据准备（若需训练）

```bash
python scripts/scan_csl.py --csl-root ./csl --report-path docs/report.md
python scripts/prepare_meta.py --csl-root ./csl --meta-dir meta --seed 3407
python scripts/extract_keypoints.py \
  --manifest meta/manifest.csv \
  --processed-root dataset_processed \
  --num-frames 32 \
  --num-workers 4
```

### 4.3 训练

```bash
python train.py \
  --manifest meta/manifest.csv \
  --processed-root dataset_processed \
  --vocab meta/vocab_gloss.json \
  --epochs 20 \
  --batch-size 64 \
  --lr 1e-3 \
  --scheduler cosine \
  --min-lr 1e-5 \
  --ckpt-dir checkpoints
```

说明：`checkpoints/best.pt` 按 **历史最佳 val top1** 维护（同一 `--ckpt-dir` 下跨次训练比较）。

### 4.4 离线推理

```bash
python infer.py \
  --input <视频文件或帧目录> \
  --checkpoint checkpoints/best.pt \
  --vocab meta/vocab_gloss.json \
  --topk 5
```

## 5. 开源说明

- 本仓库默认不包含CSL数据集、训练产物和大模型权重。
- 请根据CSL数据集许可协议下载和使用数据。

## 6. 常见问题

- `ModuleNotFoundError`：执行 `pip install -r requirements.txt`
- MediaPipe API 报错：建议固定 `mediapipe==0.10.14`
- 摄像头无画面：检查浏览器摄像头权限

## 7. License

本项目采用 [MIT License](LICENSE)。

