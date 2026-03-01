#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from slr_baseline.utils import (
    dump_json,
    iter_sample_dirs,
    read_dictionary,
    resolve_csl_root,
    to_manifest_video_path,
)


def stratified_random_split(
    rows: List[dict],
    seed: int,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> List[dict]:
    rng = random.Random(seed)
    by_label: Dict[int, List[dict]] = defaultdict(list)
    for row in rows:
        by_label[row["label_id"]].append(row)

    out = []
    for label_id, items in sorted(by_label.items()):
        shuffled = list(items)
        rng.shuffle(shuffled)

        n = len(shuffled)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        n_test = n - n_train - n_val

        # Keep split usable on small classes.
        if n >= 3:
            if n_val == 0:
                n_val = 1
                n_train -= 1
            if n_test == 0:
                n_test = 1
                n_train -= 1

        if n_train <= 0:
            raise RuntimeError(
                f"Invalid split size for label={label_id}, n={n}, got train={n_train}"
            )

        for i, row in enumerate(shuffled):
            if i < n_train:
                split = "train"
            elif i < n_train + n_val:
                split = "val"
            else:
                split = "test"
            out.append(
                {
                    "video_path": row["video_path"],
                    "label_id": row["label_id"],
                    "split": split,
                }
            )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build meta/vocab_gloss.json and meta/manifest.csv for CSL"
    )
    parser.add_argument("--csl-root", type=str, default="./csl", help="Path to CSL root")
    parser.add_argument("--meta-dir", type=str, default="meta", help="Output meta directory")
    parser.add_argument("--seed", type=int, default=3407, help="Random seed for split")
    args = parser.parse_args()

    csl_root = resolve_csl_root(args.csl_root)
    color_root = csl_root / "color_video_25000"
    dictionary_path = csl_root / "dictionary.txt"
    if not color_root.exists():
        raise FileNotFoundError(f"Missing expected dir: {color_root}")
    if not dictionary_path.exists():
        raise FileNotFoundError(f"Missing expected file: {dictionary_path}")

    dictionary = read_dictionary(dictionary_path)
    rows_raw: List[dict] = []
    used_label_ids = set()
    project_root = Path.cwd().resolve()

    for label_id, sample_dir in iter_sample_dirs(color_root):
        rows_raw.append(
            {
                "video_path": to_manifest_video_path(sample_dir, project_root),
                "label_id": label_id,
            }
        )
        used_label_ids.add(label_id)

    used_label_to_gloss: Dict[str, str] = {}
    for label_id in sorted(used_label_ids):
        dict_id = f"{label_id:06d}"
        if dict_id not in dictionary:
            raise KeyError(f"Label {label_id} maps to dict id {dict_id}, but not found in dictionary")
        used_label_to_gloss[str(label_id)] = dictionary[dict_id]

    vocab_obj = {
        "dictionary_path": str(dictionary_path.relative_to(project_root)),
        "full_vocab_size": len(dictionary),
        "used_label_count": len(used_label_to_gloss),
        "used_label_to_gloss": used_label_to_gloss,
        "full_id_to_gloss": dictionary,
    }

    split_rows = stratified_random_split(rows_raw, seed=args.seed)

    meta_dir = Path(args.meta_dir)
    meta_dir.mkdir(parents=True, exist_ok=True)
    vocab_path = meta_dir / "vocab_gloss.json"
    manifest_path = meta_dir / "manifest.csv"

    dump_json(vocab_path, vocab_obj)

    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["video_path", "label_id", "split"])
        writer.writeheader()
        writer.writerows(split_rows)

    split_stats = defaultdict(int)
    for row in split_rows:
        split_stats[row["split"]] += 1

    print(f"[prepare_meta] csl_root={csl_root}")
    print(f"[prepare_meta] rows={len(split_rows)} labels={len(used_label_to_gloss)} seed={args.seed}")
    print(
        "[prepare_meta] split counts: "
        + ", ".join(f"{k}={v}" for k, v in sorted(split_stats.items()))
    )
    print(f"[prepare_meta] vocab saved: {vocab_path.resolve()}")
    print(f"[prepare_meta] manifest saved: {manifest_path.resolve()}")
    print("[prepare_meta] manifest samples:")
    for row in split_rows[:5]:
        print(f"  {row['video_path']},{row['label_id']},{row['split']}")


if __name__ == "__main__":
    main()
