#!/usr/bin/env python3
from __future__ import annotations

import argparse
import multiprocessing as mproc
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple

try:
    import cv2
    import mediapipe as mp
    import numpy as np
    from tqdm import tqdm
except ImportError as exc:
    raise SystemExit("Missing dependencies for extract_keypoints.py. Run: pip install -r requirements.txt") from exc

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR if (THIS_DIR / "src").exists() else THIS_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from slr_baseline.data import load_manifest
from slr_baseline.keypoints import extract_frame_keypoints, load_frames_from_source
from slr_baseline.utils import sample_frame_indices, set_seed, to_npz_path_for_row


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


def _process_rows(
    row_items: List[Tuple[str, int, str]],
    processed_root: str,
    num_frames: int,
    overwrite: bool,
    worker_name: str,
    show_progress: bool,
    verbose: bool,
) -> Tuple[int, int]:
    cv2.setNumThreads(0)
    processed_root_path = Path(processed_root)

    mp_hands, mp_pose = resolve_mediapipe_apis()
    ok_count = 0
    skip_count = 0

    iterator = tqdm(row_items, desc=f"Extracting-{worker_name}") if show_progress else row_items
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
        for idx, (video_path, label_id, split) in enumerate(iterator, start=1):
            try:
                source_path = Path(video_path)
                if not source_path.is_absolute():
                    source_path = (Path.cwd() / source_path).resolve()
                if not source_path.exists():
                    raise FileNotFoundError(
                        f"source path not found: {source_path} (from manifest video_path={video_path})"
                    )

                out_path = to_npz_path_for_row(processed_root_path, video_path, split)
                out_path.parent.mkdir(parents=True, exist_ok=True)

                if out_path.exists() and not overwrite:
                    skip_count += 1
                    continue

                frames = load_frames_from_source(source_path)
                if len(frames) == 0:
                    raise RuntimeError(f"No frames found in source: {source_path}")

                sample_idx = sample_frame_indices(len(frames), num_frames)
                keypoints = np.zeros((num_frames, 55, 4), dtype=np.float32)

                for t, fi in enumerate(sample_idx):
                    frame = frames[int(fi)]
                    if frame is None:
                        raise RuntimeError(
                            f"Decoded empty frame at index={fi} from source={source_path}"
                        )
                    keypoints[t] = extract_frame_keypoints(frame, hands, pose)

                np.savez_compressed(
                    out_path,
                    keypoints=keypoints,
                    label_id=np.int32(label_id),
                    split=np.array(split),
                    video_path=np.array(video_path),
                    sampled_indices=sample_idx,
                )

                if show_progress and verbose and ok_count < 3:
                    print(
                        f"[extract:{worker_name}] sample#{idx} path={video_path} "
                        f"split={split} label={label_id} keypoints_shape={keypoints.shape} "
                        f"out={out_path}"
                    )
                ok_count += 1
            except Exception as exc:
                raise RuntimeError(
                    f"[{worker_name}] Failed at local row #{idx}: "
                    f"video_path={video_path}, label_id={label_id}, split={split}"
                ) from exc

    if verbose:
        print(f"[extract:{worker_name}] done. saved={ok_count}, skipped={skip_count}")
    return ok_count, skip_count


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extract MediaPipe keypoints (Hands 42 + upper-body Pose 13) "
            "to fixed-length T=32 npz files"
        )
    )
    parser.add_argument("--manifest", type=str, default="meta/manifest.csv")
    parser.add_argument("--processed-root", type=str, default="dataset_processed")
    parser.add_argument("--num-frames", type=int, default=32, help="Fixed temporal length T")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--max-samples", type=int, default=0, help="0 means all")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing npz")
    parser.add_argument("--num-workers", type=int, default=1, help="Parallel extraction processes")
    parser.add_argument("--num-shards", type=int, default=1, help="Total shard count")
    parser.add_argument("--shard-id", type=int, default=0, help="Current shard id [0, num_shards)")
    parser.add_argument("--chunk-size", type=int, default=64, help="Task chunk size for multi-worker mode")
    args = parser.parse_args()

    if args.num_frames <= 0:
        raise ValueError(f"--num-frames must be > 0, got {args.num_frames}")
    if args.num_workers <= 0:
        raise ValueError(f"--num-workers must be > 0, got {args.num_workers}")
    if args.num_shards <= 0:
        raise ValueError(f"--num-shards must be > 0, got {args.num_shards}")
    if not (0 <= args.shard_id < args.num_shards):
        raise ValueError(
            f"--shard-id must be in [0, {args.num_shards}), got {args.shard_id}"
        )
    if args.chunk_size <= 0:
        raise ValueError(f"--chunk-size must be > 0, got {args.chunk_size}")

    set_seed(args.seed)
    rows = load_manifest(args.manifest)
    row_items: List[Tuple[str, int, str]] = [
        (r.video_path, int(r.label_id), r.split) for r in rows
    ]
    if args.num_shards > 1:
        row_items = [
            row for i, row in enumerate(row_items) if i % args.num_shards == args.shard_id
        ]
    if args.max_samples > 0:
        row_items = row_items[: args.max_samples]

    if not row_items:
        raise RuntimeError("No rows loaded from manifest")

    processed_root = Path(args.processed_root)
    print(
        f"[extract] manifest={Path(args.manifest).resolve()} rows={len(row_items)} "
        f"processed_root={processed_root.resolve()} T={args.num_frames} "
        f"workers={args.num_workers} shard={args.shard_id}/{args.num_shards}"
    )

    mp_hands, mp_pose = resolve_mediapipe_apis()
    print(
        f"[extract] mediapipe={getattr(mp, '__version__', 'unknown')} "
        f"hands_api={getattr(mp_hands, '__name__', str(mp_hands))} "
        f"pose_api={getattr(mp_pose, '__name__', str(mp_pose))}"
    )

    if args.num_workers == 1:
        ok_count, skip_count = _process_rows(
            row_items=row_items,
            processed_root=str(processed_root),
            num_frames=args.num_frames,
            overwrite=args.overwrite,
            worker_name="w0",
            show_progress=True,
            verbose=True,
        )
    else:
        row_chunks = [
            row_items[i : i + args.chunk_size]
            for i in range(0, len(row_items), args.chunk_size)
        ]
        print(
            f"[extract] using {args.num_workers} worker processes, "
            f"chunks={len(row_chunks)}, chunk_size={args.chunk_size}"
        )

        ok_count = 0
        skip_count = 0
        try:
            ctx = mproc.get_context("spawn")
            with ProcessPoolExecutor(
                max_workers=args.num_workers,
                mp_context=ctx,
            ) as executor:
                futures = {}
                for ti, subset in enumerate(row_chunks):
                    fut = executor.submit(
                        _process_rows,
                        subset,
                        str(processed_root),
                        args.num_frames,
                        args.overwrite,
                        f"task{ti}",
                        False,
                        False,
                    )
                    futures[fut] = len(subset)

                with tqdm(total=len(row_items), desc="Extracting-multi") as pbar:
                    for fut in as_completed(futures):
                        chunk_len = futures[fut]
                        try:
                            saved_i, skipped_i = fut.result()
                            ok_count += saved_i
                            skip_count += skipped_i
                            pbar.update(chunk_len)
                        except Exception as exc:
                            raise RuntimeError("A worker task failed") from exc
        except PermissionError as exc:
            raise RuntimeError(
                "Failed to create multiprocessing workers due system semaphore permission. "
                "Use `--num-workers 1` and run sharded extraction in parallel terminals, e.g.\n"
                "  terminal1: --num-workers 1 --num-shards 4 --shard-id 0\n"
                "  terminal2: --num-workers 1 --num-shards 4 --shard-id 1\n"
                "  terminal3: --num-workers 1 --num-shards 4 --shard-id 2\n"
                "  terminal4: --num-workers 1 --num-shards 4 --shard-id 3"
            ) from exc

    print(
        f"[extract] done. saved={ok_count}, skipped={skip_count}, "
        f"processed_root={processed_root.resolve()}"
    )


if __name__ == "__main__":
    cv2.setNumThreads(0)
    main()
