# CSL Hands+Pose Baseline (PyTorch)

目标：把 CSL 数据集做成可训练的 Hands + Pose（上半身）分类 baseline，并提供可复现命令。

## 目录结构

- `scan_csl.py`: 扫描 `./csl`，输出 `report.md`
- `prepare_meta.py`: 生成 `meta/vocab_gloss.json` 与 `meta/manifest.csv`
- `extract_keypoints.py`: MediaPipe 提取关键点，输出 `dataset_processed/{split}/*.npz`
- `visualize_samples.py`: 随机抽样可视化，输出 `debug_vis/*.mp4`
- `train.py`: 训练 baseline（Linear -> BiLSTM -> Pool -> Linear）
- `realtime_demo.py`: 摄像头 1 秒滑窗推理 demo
- `slr_baseline/`: 公共模块（模型、数据读取、关键点工具）

## 数据准备

默认会读取 `./csl`。本项目中已创建软链接：`csl -> dataset/csl`。

请确认以下路径存在：

- `csl/dictionary.txt`
- `csl/color_video_25000/`

## 安装依赖

```bash
pip install -r requirements.txt
```

说明：当前基线依赖 `mediapipe==0.10.14`（使用 `solutions` API）。

## 1) 扫描数据并产出报告

```bash
python scan_csl.py --csl-root ./csl --report-path report.md
```

输出：

- `report.md`（含样例行、标注文件、划分文件检查结果）

## 2) 生成词表与清单（可复现 split）

```bash
python prepare_meta.py --csl-root ./csl --meta-dir meta --seed 3407
```

输出：

- `meta/vocab_gloss.json`
- `meta/manifest.csv`（列：`video_path,label_id,split`）

说明：如无官方 split，脚本按分层随机 `8:1:1` 生成 `train/val/test`，固定 `seed` 可复现。

## 3) 提取 Hands+Pose 关键点

```bash
python extract_keypoints.py \
  --manifest meta/manifest.csv \
  --processed-root dataset_processed \
  --num-frames 32 \
  --seed 3407 \
  --num-workers 4
```

输出 shape：`[T=32, 55, 4]`

- 55 点 = Hands 42 点（左21 + 右21） + Pose 上半身 13 点
- 每点 4 维 = `[x, y, z, vis]`
- 缺失点补 0
- 支持多进程提取：`--num-workers N`
- 支持分片：`--num-shards K --shard-id I`（用于多机/多任务并行）

## 4) 可视化抽样检查（20个）

```bash
python visualize_samples.py \
  --manifest meta/manifest.csv \
  --processed-root dataset_processed \
  --output-dir debug_vis \
  --num-samples 20 \
  --seed 3407
```

输出：`debug_vis/*.mp4`

## 5) 训练 baseline

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
  --seed 3407 \
  --ckpt-dir checkpoints
```

训练输出：

- `top1 / top5`
- 最优 checkpoint：`checkpoints/best.pt`
- 默认启用：关键点归一化 + 速度特征（可用 `--no-normalize` / `--no-velocity` 关闭）

## 6) 实时 demo

```bash
python realtime_demo.py \
  --checkpoint checkpoints/best.pt \
  --vocab meta/vocab_gloss.json \
  --camera-id 0 \
  --window-seconds 1.0
```

操作：按 `q` 退出。

## 常见错误与排查

- `CSL root not found`：检查 `--csl-root` 是否正确，或确认软链接 `./csl` 存在。
- `Missing processed npz files`：先运行 `extract_keypoints.py`。
- `Failed to open camera`：检查摄像头权限或改 `--camera-id`。
- 中文词条显示异常：系统缺少 CJK 字体，仍可通过 label id 推理。
- `Cannot access MediaPipe Hands/Pose APIs`：执行
  `pip uninstall -y mediapipe mediapipe-nightly && pip install mediapipe==0.10.14`。
