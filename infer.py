#!/usr/bin/env python3
"""
软件名称：基于深度学习的离散手语识别系统
Software Name: Deep Learning Based Discrete Sign Language Recognition System
版本号：V1.0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

try:
    import cv2
    import mediapipe as mp
    import numpy as np
    import torch
except ImportError as exc:
    raise SystemExit("Missing dependencies for infer.py. Run: pip install -r requirements.txt") from exc

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from slr_baseline.features import build_sequence_features
from slr_baseline.keypoints import extract_frame_keypoints, load_frames_from_source
from slr_baseline.model import SignBiLSTMBaseline
from slr_baseline.utils import sample_frame_indices


def resolve_mediapipe_apis():
    """Return (hands_module, pose_module) across MediaPipe packaging variants."""
    try:
        return mp.solutions.hands, mp.solutions.pose
    except Exception:
        pass

    try:
        from mediapipe.python.solutions import hands as mp_hands  # type: ignore
        from mediapipe.python.solutions import pose as mp_pose  # type: ignore

        return mp_hands, mp_pose
    except Exception as exc:
        mp_ver = getattr(mp, "__version__", "unknown")
        raise RuntimeError(
            "Cannot access MediaPipe Hands/Pose APIs. "
            f"Detected mediapipe={mp_ver}. This project requires the legacy solutions API.\n"
            "Fix with:\n"
            "  pip uninstall -y mediapipe mediapipe-nightly\n"
            "  pip install mediapipe==0.10.14"
        ) from exc


def pick_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device_arg)


def load_vocab(vocab_path: Path) -> Dict[str, str]:
    if not vocab_path.exists():
        raise FileNotFoundError(f"Vocab file not found: {vocab_path}")
    with vocab_path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    mapping = obj.get("used_label_to_gloss")
    if not isinstance(mapping, dict):
        raise ValueError("vocab_gloss.json missing `used_label_to_gloss`")
    return mapping


def ensure_ckpt_config(ckpt: dict) -> dict:
    cfg = ckpt.get("config", {})
    required = ["num_classes", "input_dim", "proj_dim", "hidden_size", "num_layers", "dropout", "num_frames"]
    miss = [k for k in required if k not in cfg]
    if miss:
        raise KeyError(f"Checkpoint config missing keys: {miss}")
    return cfg


def get_video_fps(source_path: Path) -> Optional[float]:
    if source_path.is_dir():
        return None
    cap = cv2.VideoCapture(str(source_path))
    if not cap.isOpened():
        return None
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    if fps <= 1e-3:
        return None
    return fps


def build_windows(
    num_frames: int,
    window_size: int,
    stride: int,
    single_clip: bool,
) -> List[Tuple[int, int]]:
    if num_frames <= 0:
        raise ValueError(f"num_frames must be > 0, got {num_frames}")
    if single_clip:
        return [(0, num_frames)]
    if window_size <= 0:
        raise ValueError(f"window_size must be > 0, got {window_size}")
    if stride <= 0:
        raise ValueError(f"stride must be > 0, got {stride}")

    if num_frames <= window_size:
        return [(0, num_frames)]

    starts = list(range(0, num_frames - window_size + 1, stride))
    last_start = num_frames - window_size
    if starts[-1] != last_start:
        starts.append(last_start)
    return [(s, s + window_size) for s in starts]


def infer_sequence(
    model: torch.nn.Module,
    keypoints_t554: np.ndarray,
    input_dim: int,
    use_normalize: bool,
    use_velocity: bool,
    device: torch.device,
    topk: int,
    vocab: Dict[str, str],
) -> List[dict]:
    seq_feat = build_sequence_features(
        keypoints_t554,
        normalize=use_normalize,
        use_velocity=use_velocity,
        flatten=True,
    )
    if seq_feat.shape[1] != input_dim:
        raise RuntimeError(
            f"Feature dim mismatch: got {seq_feat.shape[1]}, checkpoint expects {input_dim}"
        )

    x = torch.from_numpy(seq_feat).unsqueeze(0).to(device=device, dtype=torch.float32)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]

    k = min(topk, int(probs.shape[0]))
    top_vals, top_ids = torch.topk(probs, k=k, dim=0, largest=True, sorted=True)
    preds: List[dict] = []
    for rank, (label_id, prob) in enumerate(zip(top_ids.tolist(), top_vals.tolist()), start=1):
        gloss = vocab.get(str(label_id), f"id:{label_id}")
        preds.append(
            {
                "rank": rank,
                "label_id": int(label_id),
                "gloss": gloss,
                "prob": float(prob),
            }
        )
    return preds


def merge_window_predictions(window_preds: Sequence[dict]) -> List[dict]:
    if not window_preds:
        return []

    merged: List[dict] = []
    cur = None
    for w in window_preds:
        top1 = w["top1"]
        if cur is None:
            cur = {
                "label_id": int(top1["label_id"]),
                "gloss": str(top1["gloss"]),
                "start_frame": int(w["start_frame"]),
                "end_frame": int(w["end_frame"]),
                "max_prob": float(top1["prob"]),
                "sum_prob": float(top1["prob"]),
                "num_windows": 1,
            }
            continue

        same_label = int(top1["label_id"]) == int(cur["label_id"])
        overlap_or_adjacent = int(w["start_frame"]) <= int(cur["end_frame"])
        if same_label and overlap_or_adjacent:
            cur["end_frame"] = max(int(cur["end_frame"]), int(w["end_frame"]))
            cur["max_prob"] = max(float(cur["max_prob"]), float(top1["prob"]))
            cur["sum_prob"] = float(cur["sum_prob"]) + float(top1["prob"])
            cur["num_windows"] = int(cur["num_windows"]) + 1
        else:
            cur["avg_prob"] = float(cur["sum_prob"]) / max(int(cur["num_windows"]), 1)
            del cur["sum_prob"]
            merged.append(cur)
            cur = {
                "label_id": int(top1["label_id"]),
                "gloss": str(top1["gloss"]),
                "start_frame": int(w["start_frame"]),
                "end_frame": int(w["end_frame"]),
                "max_prob": float(top1["prob"]),
                "sum_prob": float(top1["prob"]),
                "num_windows": 1,
            }

    cur["avg_prob"] = float(cur["sum_prob"]) / max(int(cur["num_windows"]), 1)
    del cur["sum_prob"]
    merged.append(cur)
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline inference for CSL Hands+Pose baseline")
    parser.add_argument("--input", type=str, required=True, help="Video file path or frame directory path")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pt")
    parser.add_argument("--vocab", type=str, default="meta/vocab_gloss.json")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num-frames", type=int, default=0, help="Override temporal length T; 0 means use checkpoint config")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--window-size", type=int, default=0, help="Sliding window length on source frames; 0 means use model T")
    parser.add_argument("--stride", type=int, default=0, help="Sliding window stride on source frames; 0 means window_size//2")
    parser.add_argument("--single-clip", action="store_true", help="Disable sliding window and output only one prediction for full input")
    parser.add_argument("--save-keypoints", type=str, default="", help="Optional output npz path for extracted [T,55,4] keypoints")
    parser.add_argument("--output-json", type=str, default="", help="Optional output JSON path for prediction results")
    args = parser.parse_args()

    if args.topk <= 0:
        raise ValueError(f"--topk must be > 0, got {args.topk}")
    if args.num_frames < 0:
        raise ValueError(f"--num-frames must be >= 0, got {args.num_frames}")
    if args.window_size < 0:
        raise ValueError(f"--window-size must be >= 0, got {args.window_size}")
    if args.stride < 0:
        raise ValueError(f"--stride must be >= 0, got {args.stride}")

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input source not found: {input_path}")

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    device = pick_device(args.device)
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ensure_ckpt_config(ckpt)

    model = SignBiLSTMBaseline(
        num_classes=int(cfg["num_classes"]),
        input_dim=int(cfg["input_dim"]),
        proj_dim=int(cfg["proj_dim"]),
        hidden_size=int(cfg["hidden_size"]),
        num_layers=int(cfg["num_layers"]),
        dropout=float(cfg["dropout"]),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    model_t = int(cfg["num_frames"])
    target_t = int(args.num_frames) if args.num_frames > 0 else model_t
    if target_t <= 0:
        raise ValueError(f"Invalid temporal length T={target_t}")

    input_dim = int(cfg["input_dim"])
    use_normalize = bool(cfg.get("normalize", False))
    use_velocity = bool(cfg.get("use_velocity", False))

    vocab = load_vocab(Path(args.vocab))
    frames = load_frames_from_source(input_path)
    num_source_frames = len(frames)
    window_size = int(args.window_size) if args.window_size > 0 else target_t
    stride = int(args.stride) if args.stride > 0 else max(1, window_size // 2)
    windows = build_windows(
        num_frames=num_source_frames,
        window_size=window_size,
        stride=stride,
        single_clip=bool(args.single_clip),
    )
    fps = get_video_fps(input_path)

    mp_hands, mp_pose = resolve_mediapipe_apis()
    print(
        f"[infer] source={input_path.resolve()} decoded_frames={num_source_frames} "
        f"sampled_T={target_t} windows={len(windows)} window_size={window_size} "
        f"stride={stride} single_clip={args.single_clip} "
        f"device={device} normalize={use_normalize} velocity={use_velocity}"
    )
    print(
        f"[infer] mediapipe={getattr(mp, '__version__', 'unknown')} "
        f"hands_api={getattr(mp_hands, '__name__', str(mp_hands))} "
        f"pose_api={getattr(mp_pose, '__name__', str(mp_pose))}"
    )
    if fps is not None:
        print(f"[infer] source_fps={fps:.3f}")

    full_keypoints = np.zeros((num_source_frames, 55, 4), dtype=np.float32)
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as hands, mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        for fi, frame in enumerate(frames):
            full_keypoints[fi] = extract_frame_keypoints(frame, hands, pose)

    window_preds: List[dict] = []
    sampled_indices_per_window: List[List[int]] = []
    sampled_keypoints_per_window: List[np.ndarray] = []
    for wid, (start, end) in enumerate(windows):
        win_len = end - start
        idx_local = sample_frame_indices(win_len, target_t)
        idx_global = [int(start + int(i)) for i in idx_local]
        clip_keypoints = full_keypoints[np.asarray(idx_global, dtype=np.int32)]

        preds = infer_sequence(
            model=model,
            keypoints_t554=clip_keypoints,
            input_dim=input_dim,
            use_normalize=use_normalize,
            use_velocity=use_velocity,
            device=device,
            topk=args.topk,
            vocab=vocab,
        )

        rec = {
            "window_id": int(wid),
            "start_frame": int(start),
            "end_frame": int(end),
            "sampled_indices": idx_global,
            "top1": preds[0],
            "topk": preds,
        }
        if fps is not None:
            rec["start_time_sec"] = float(start / fps)
            rec["end_time_sec"] = float(end / fps)
        window_preds.append(rec)
        sampled_indices_per_window.append(idx_global)
        if args.save_keypoints:
            sampled_keypoints_per_window.append(clip_keypoints)

    if args.save_keypoints:
        out_npz = Path(args.save_keypoints)
        out_npz.parent.mkdir(parents=True, exist_ok=True)
        win_ranges = np.asarray(windows, dtype=np.int32)
        sampled_idx = np.asarray(sampled_indices_per_window, dtype=np.int32)
        sampled_kpts = np.asarray(sampled_keypoints_per_window, dtype=np.float32)
        np.savez_compressed(
            out_npz,
            full_keypoints=full_keypoints,
            sampled_keypoints=sampled_kpts,
            sampled_indices=sampled_idx,
            window_ranges=win_ranges,
            source_path=np.array(str(input_path)),
            num_source_frames=np.int32(num_source_frames),
        )
        print(f"[infer] saved keypoints: {out_npz.resolve()}")

    print("[infer] window predictions:")
    for w in window_preds:
        top1 = w["top1"]
        if "start_time_sec" in w and "end_time_sec" in w:
            print(
                f"  w{w['window_id']:03d} frame=[{w['start_frame']},{w['end_frame']}) "
                f"time=[{w['start_time_sec']:.2f},{w['end_time_sec']:.2f}) "
                f"-> {top1['label_id']} {top1['gloss']} ({top1['prob']:.4f})"
            )
        else:
            print(
                f"  w{w['window_id']:03d} frame=[{w['start_frame']},{w['end_frame']}) "
                f"-> {top1['label_id']} {top1['gloss']} ({top1['prob']:.4f})"
            )

    merged_segments = merge_window_predictions(window_preds)
    print("[infer] merged segments:")
    for i, s in enumerate(merged_segments, start=1):
        if fps is not None:
            st = float(s["start_frame"]) / fps
            et = float(s["end_frame"]) / fps
            print(
                f"  seg{i:03d} frame=[{s['start_frame']},{s['end_frame']}) "
                f"time=[{st:.2f},{et:.2f}) -> {s['label_id']} {s['gloss']} "
                f"(avg={s['avg_prob']:.4f}, max={s['max_prob']:.4f}, windows={s['num_windows']})"
            )
        else:
            print(
                f"  seg{i:03d} frame=[{s['start_frame']},{s['end_frame']}) "
                f"-> {s['label_id']} {s['gloss']} "
                f"(avg={s['avg_prob']:.4f}, max={s['max_prob']:.4f}, windows={s['num_windows']})"
            )

    if args.output_json:
        out_json = Path(args.output_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "input_path": str(input_path.resolve()),
            "checkpoint": str(ckpt_path.resolve()),
            "decoded_frames": int(num_source_frames),
            "fps": fps,
            "sampled_num_frames": int(target_t),
            "window_size": int(window_size),
            "stride": int(stride),
            "single_clip": bool(args.single_clip),
            "normalize": use_normalize,
            "use_velocity": use_velocity,
            "window_predictions": window_preds,
            "merged_segments": merged_segments,
        }
        with out_json.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"[infer] saved predictions: {out_json.resolve()}")


if __name__ == "__main__":
    cv2.setNumThreads(0)
    main()
