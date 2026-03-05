from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset

from .features import build_sequence_features
from .utils import to_npz_path_for_row


@dataclass
class ManifestRow:
    video_path: str
    label_id: int
    split: str


def load_manifest(manifest_csv: str | Path) -> List[ManifestRow]:
    manifest_path = Path(manifest_csv)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    rows: List[ManifestRow] = []
    with manifest_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"video_path", "label_id", "split"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Manifest missing columns: {sorted(missing)}")

        for i, row in enumerate(reader, start=2):
            try:
                rows.append(
                    ManifestRow(
                        video_path=row["video_path"],
                        label_id=int(row["label_id"]),
                        split=row["split"],
                    )
                )
            except Exception as exc:
                raise ValueError(f"Bad manifest row at line {i}: {row}") from exc
    return rows


class KeypointNPZDataset(Dataset):
    def __init__(
        self,
        manifest_csv: str | Path,
        processed_root: str | Path,
        split: str,
        strict: bool = True,
        normalize: bool = True,
        use_velocity: bool = True,
    ) -> None:
        self.manifest_csv = Path(manifest_csv)
        self.processed_root = Path(processed_root)
        self.split = split
        self.strict = strict
        self.normalize = normalize
        self.use_velocity = use_velocity

        all_rows = load_manifest(self.manifest_csv)
        self.rows = [r for r in all_rows if r.split == split]
        if not self.rows:
            raise ValueError(
                f"No rows found for split='{split}' in manifest: {self.manifest_csv}"
            )

        self.samples = []
        missing_paths = []
        for row in self.rows:
            npz_path = to_npz_path_for_row(self.processed_root, row.video_path, row.split)
            if not npz_path.exists():
                missing_paths.append(npz_path)
                continue
            self.samples.append((row, npz_path))

        if strict and missing_paths:
            preview = "\n".join(str(p) for p in missing_paths[:5])
            raise FileNotFoundError(
                "Missing processed npz files. Run scripts/extract_keypoints.py first. "
                f"Examples:\n{preview}"
            )

        if not self.samples:
            raise RuntimeError(
                f"No usable samples for split='{split}'. "
                "Check processed_root or disable strict mode."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        row, npz_path = self.samples[idx]
        if not npz_path.exists():
            raise FileNotFoundError(f"Processed sample not found: {npz_path}")

        arr = np.load(npz_path)
        if "keypoints" not in arr:
            raise KeyError(f"npz missing 'keypoints': {npz_path}")

        keypoints = arr["keypoints"].astype(np.float32)
        if keypoints.ndim != 3 or keypoints.shape[1:] != (55, 4):
            raise ValueError(
                f"Unexpected keypoint shape {keypoints.shape} in {npz_path}, expected [T,55,4]"
            )

        features = build_sequence_features(
            keypoints,
            normalize=self.normalize,
            use_velocity=self.use_velocity,
            flatten=True,
        )
        x = torch.from_numpy(features)
        y = torch.tensor(row.label_id, dtype=torch.long)
        return x, y, row.video_path
