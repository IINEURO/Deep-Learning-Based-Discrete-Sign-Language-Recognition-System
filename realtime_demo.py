#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path
from typing import Dict, Optional

try:
    import cv2
    import mediapipe as mp
    import numpy as np
    import torch
    from PIL import Image, ImageDraw, ImageFont
except ImportError as exc:
    raise SystemExit("Missing dependencies for realtime_demo.py. Run: pip install -r requirements.txt") from exc

from slr_baseline.features import build_sequence_features
from slr_baseline.keypoints import draw_keypoints, extract_frame_keypoints
from slr_baseline.model import SignBiLSTMBaseline


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


def load_font(size: int = 26) -> Optional[ImageFont.FreeTypeFont]:
    candidates = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/arphic/ukai.ttc",
    ]
    for p in candidates:
        pp = Path(p)
        if pp.exists():
            try:
                return ImageFont.truetype(str(pp), size=size)
            except Exception:
                continue
    return None


def draw_text(frame_bgr: np.ndarray, text: str, org=(10, 10), font: Optional[ImageFont.FreeTypeFont] = None) -> np.ndarray:
    if font is None:
        x, y = org
        cv2.putText(
            frame_bgr,
            text,
            (x, y + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return frame_bgr

    img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)
    draw.text(org, text, font=font, fill=(255, 255, 255))
    out = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Realtime CSL recognition demo")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pt")
    parser.add_argument("--vocab", type=str, default="meta/vocab_gloss.json")
    parser.add_argument("--camera-id", type=int, default=0)
    parser.add_argument("--window-seconds", type=float, default=1.0)
    parser.add_argument("--target-fps", type=int, default=30)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    device = pick_device(args.device)
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt.get("config", {})
    required = ["num_classes", "input_dim", "proj_dim", "hidden_size", "num_layers", "dropout", "num_frames"]
    miss = [k for k in required if k not in cfg]
    if miss:
        raise KeyError(f"Checkpoint config missing keys: {miss}")

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
    if model_t <= 0:
        raise ValueError(f"Invalid num_frames from checkpoint: {model_t}")
    input_dim = int(cfg["input_dim"])
    use_normalize = bool(cfg.get("normalize", False))
    use_velocity = bool(cfg.get("use_velocity", False))

    vocab = load_vocab(Path(args.vocab))
    cjk_font = load_font(size=24)
    if cjk_font is None:
        print("[demo] warning: CJK font not found, Chinese rendering may be degraded")

    window_size = max(2, int(args.window_seconds * args.target_fps))
    buf = deque(maxlen=window_size)

    print(
        f"[demo] checkpoint={ckpt_path.resolve()} device={device} "
        f"model_T={model_t} input_dim={input_dim} window_size={window_size} "
        f"normalize={use_normalize} velocity={use_velocity} camera_id={args.camera_id}"
    )

    cap = cv2.VideoCapture(args.camera_id)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera id={args.camera_id}")

    mp_hands, mp_pose = resolve_mediapipe_apis()
    print(
        f"[demo] mediapipe={getattr(mp, '__version__', 'unknown')} "
        f"hands_api={getattr(mp_hands, '__name__', str(mp_hands))} "
        f"pose_api={getattr(mp_pose, '__name__', str(mp_pose))}"
    )

    pred_label = -1
    pred_prob = 0.0

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
        while True:
            ret, frame = cap.read()
            if not ret:
                raise RuntimeError("Failed to read frame from camera")

            kpt = extract_frame_keypoints(frame, hands, pose)
            buf.append(kpt)

            vis = draw_keypoints(frame, kpt)

            if len(buf) >= 2:
                seq_np = np.stack(list(buf), axis=0)
                idx = np.linspace(0, seq_np.shape[0] - 1, model_t).astype(np.int32)
                seq_fixed = seq_np[idx]
                seq_feat = build_sequence_features(
                    seq_fixed,
                    normalize=use_normalize,
                    use_velocity=use_velocity,
                    flatten=True,
                )
                if seq_feat.shape[1] != input_dim:
                    raise RuntimeError(
                        f"Feature dim mismatch at runtime: got {seq_feat.shape[1]}, "
                        f"checkpoint expects {input_dim}"
                    )

                x = torch.from_numpy(seq_feat).unsqueeze(0).to(device=device, dtype=torch.float32)
                with torch.no_grad():
                    logits = model(x)
                    probs = torch.softmax(logits, dim=1)[0]
                    pred_label = int(torch.argmax(probs).item())
                    pred_prob = float(probs[pred_label].item())

            gloss = vocab.get(str(pred_label), f"id:{pred_label}")
            line1 = f"Pred: {pred_label} {gloss} ({pred_prob:.2f})"
            line2 = f"Window: {len(buf)}/{window_size} frames"
            vis = draw_text(vis, line1, org=(10, 10), font=cjk_font)
            vis = draw_text(vis, line2, org=(10, 44), font=cjk_font)

            cv2.imshow("CSL Realtime Demo (press q to quit)", vis)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    cv2.setNumThreads(0)
    main()
