from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

try:
    import numpy as np
except ImportError:  # Optional for lightweight scripts like scan/prepare.
    np = None

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def resolve_csl_root(csl_root: str | Path) -> Path:
    candidates = [Path(csl_root), Path("./dataset/csl"), Path("./csl")]
    for c in candidates:
        if c.exists() and c.is_dir():
            return c.resolve()
    raise FileNotFoundError(
        "CSL root not found. Tried: " + ", ".join(str(c) for c in candidates)
    )


def read_dictionary(dictionary_path: str | Path) -> Dict[str, str]:
    path = Path(dictionary_path)
    if not path.exists():
        raise FileNotFoundError(f"dictionary.txt not found: {path}")

    mapping: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                raise ValueError(f"Bad dictionary line {ln}: {line}")
            key = parts[0].strip()
            gloss = parts[1].strip()
            mapping[key] = gloss
    return mapping


def discover_label_dirs(color_root: str | Path) -> List[Path]:
    root = Path(color_root)
    if not root.exists():
        raise FileNotFoundError(f"color_video_25000 dir not found: {root}")
    label_dirs = sorted([p for p in root.iterdir() if p.is_dir() and p.name.isdigit()])
    if not label_dirs:
        raise RuntimeError(f"No label dirs found in: {root}")
    return label_dirs


def iter_sample_dirs(color_root: str | Path) -> Iterable[Tuple[int, Path]]:
    for label_dir in discover_label_dirs(color_root):
        label_id = int(label_dir.name)
        for sample_dir in sorted([p for p in label_dir.iterdir() if p.is_dir()]):
            yield label_id, sample_dir


def list_frames_from_dir(sample_dir: str | Path) -> List[Path]:
    d = Path(sample_dir)
    if not d.exists() or not d.is_dir():
        raise FileNotFoundError(f"Sample dir not found: {d}")
    frames = sorted([p for p in d.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS])
    return frames


def sample_frame_indices(num_frames: int, t: int):
    if num_frames <= 0:
        raise ValueError(f"num_frames must be > 0, got {num_frames}")
    if t <= 0:
        raise ValueError(f"t must be > 0, got {t}")
    if np is not None:
        return np.linspace(0, num_frames - 1, t).astype(np.int32)

    # Fallback without numpy.
    if t == 1:
        return [0]
    step = (num_frames - 1) / (t - 1)
    return [int(round(i * step)) for i in range(t)]


def to_manifest_video_path(path: Path, base_dir: Path) -> str:
    return str(path.resolve().relative_to(base_dir.resolve()))


def stable_sample_key(video_path: str) -> str:
    return hashlib.sha1(video_path.encode("utf-8")).hexdigest()[:16]


def to_npz_path_for_row(processed_root: str | Path, video_path: str, split: str) -> Path:
    key = stable_sample_key(video_path)
    return Path(processed_root) / split / f"{key}.npz"


def dump_json(path: str | Path, obj) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
