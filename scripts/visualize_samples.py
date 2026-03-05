#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import cv2
    import numpy as np
except ImportError as exc:
    raise SystemExit("Missing dependencies for visualize_samples.py. Run: pip install -r requirements.txt") from exc

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR if (THIS_DIR / "src").exists() else THIS_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from slr_baseline.data import load_manifest
from slr_baseline.keypoints import draw_keypoints, load_frames_from_source
from slr_baseline.utils import set_seed, to_npz_path_for_row


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize random samples by overlaying keypoint skeletons to frames"
    )
    parser.add_argument("--manifest", type=str, default="meta/manifest.csv")
    parser.add_argument("--processed-root", type=str, default="dataset_processed")
    parser.add_argument("--output-dir", type=str, default="outputs/debug_vis")
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--split", type=str, default="", help="Optional filter: train/val/test")
    args = parser.parse_args()

    if args.num_samples <= 0:
        raise ValueError(f"--num-samples must be > 0, got {args.num_samples}")

    set_seed(args.seed)
    rows = load_manifest(args.manifest)
    if args.split:
        rows = [r for r in rows if r.split == args.split]

    if not rows:
        raise RuntimeError("No rows available for visualization")

    n = min(args.num_samples, len(rows))
    rng = np.random.default_rng(args.seed)
    pick_idx = rng.choice(len(rows), size=n, replace=False)
    picked = [rows[i] for i in pick_idx]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[vis] manifest={Path(args.manifest).resolve()} picked={n}/{len(rows)} "
        f"processed_root={Path(args.processed_root).resolve()} output_dir={output_dir.resolve()}"
    )

    for i, row in enumerate(picked, start=1):
        source_path = Path(row.video_path)
        if not source_path.is_absolute():
            source_path = (Path.cwd() / source_path).resolve()
        if not source_path.exists():
            raise FileNotFoundError(f"source path not found: {source_path}")

        npz_path = to_npz_path_for_row(args.processed_root, row.video_path, row.split)
        if not npz_path.exists():
            raise FileNotFoundError(
                f"Missing keypoint npz: {npz_path}. Run scripts/extract_keypoints.py first."
            )

        npz = np.load(npz_path)
        keypoints = npz["keypoints"]
        sampled_indices = npz["sampled_indices"]

        if keypoints.ndim != 3 or keypoints.shape[1:] != (55, 4):
            raise ValueError(
                f"Unexpected keypoint shape in {npz_path}: {keypoints.shape}, expected [T,55,4]"
            )

        frames = load_frames_from_source(source_path)
        if len(frames) == 0:
            raise RuntimeError(f"No frames loaded from {source_path}")

        T = keypoints.shape[0]
        if sampled_indices.shape[0] != T:
            raise ValueError(
                f"Mismatch keypoints T={T} and sampled_indices len={sampled_indices.shape[0]} "
                f"in {npz_path}"
            )

        h, w = frames[0].shape[:2]
        out_name = f"{i:03d}_label{row.label_id}_{npz_path.stem}.mp4"
        out_path = output_dir / out_name
        writer = cv2.VideoWriter(
            str(out_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            args.fps,
            (w, h),
        )

        if not writer.isOpened():
            raise RuntimeError(f"Failed to create VideoWriter: {out_path}")

        for t in range(T):
            fi = int(sampled_indices[t])
            fi = max(0, min(fi, len(frames) - 1))
            frame = frames[fi]
            if frame.shape[:2] != (h, w):
                frame = cv2.resize(frame, (w, h))

            vis = draw_keypoints(frame, keypoints[t])
            cv2.putText(
                vis,
                f"label={row.label_id} split={row.split} t={t+1}/{T}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            writer.write(vis)

        writer.release()
        print(
            f"[vis] {i:02d}/{n} saved={out_path} "
            f"source={row.video_path} keypoints_shape={keypoints.shape}"
        )

    print(f"[vis] done. output_dir={output_dir.resolve()}")


if __name__ == "__main__":
    cv2.setNumThreads(0)
    main()
