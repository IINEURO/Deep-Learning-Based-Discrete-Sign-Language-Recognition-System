#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List

from slr_baseline.utils import discover_label_dirs, iter_sample_dirs, list_frames_from_dir, resolve_csl_root


def find_meta_files(csl_root: Path) -> tuple[List[Path], List[Path]]:
    """Find annotation/split-like files outside the huge frame directory."""
    annotation_files: List[Path] = []
    split_files: List[Path] = []
    keywords_ann = ("dict", "label", "anno", "annotation", "gloss")
    keywords_split = ("train", "val", "test", "split")

    for root, dirs, files in os.walk(csl_root):
        # Skip the huge frame tree to keep scanning fast.
        if "color_video_25000" in dirs:
            dirs.remove("color_video_25000")
        for fn in files:
            p = Path(root) / fn
            name = fn.lower()
            if any(k in name for k in keywords_ann):
                annotation_files.append(p)
            if any(k in name for k in keywords_split):
                split_files.append(p)

    annotation_files = sorted(set(annotation_files))
    split_files = sorted(set(split_files))
    return annotation_files, split_files


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan CSL dataset and generate report.md")
    parser.add_argument("--csl-root", type=str, default="./csl", help="Path to CSL root")
    parser.add_argument("--report-path", type=str, default="report.md", help="Output markdown path")
    parser.add_argument("--preview", type=int, default=10, help="Number of sample rows to show")
    args = parser.parse_args()

    csl_root = resolve_csl_root(args.csl_root)
    color_root = csl_root / "color_video_25000"
    dictionary_path = csl_root / "dictionary.txt"

    if not color_root.exists():
        raise FileNotFoundError(f"Missing expected dir: {color_root}")
    if not dictionary_path.exists():
        raise FileNotFoundError(f"Missing expected file: {dictionary_path}")

    label_dirs = discover_label_dirs(color_root)
    sample_rows = []
    total_samples = 0
    for label_id, sample_dir in iter_sample_dirs(color_root):
        total_samples += 1
        if len(sample_rows) < args.preview:
            frames = list_frames_from_dir(sample_dir)
            sample_rows.append(
                {
                    "video_path": str(sample_dir.relative_to(Path.cwd())),
                    "label_id": label_id,
                    "frame_count": len(frames),
                }
            )

    annotation_files, split_files = find_meta_files(csl_root)

    report_lines: List[str] = []
    report_lines.append("# CSL Dataset Scan Report")
    report_lines.append("")
    report_lines.append(f"- Resolved CSL root: `{csl_root}`")
    report_lines.append(f"- Dictionary file: `{dictionary_path}`")
    report_lines.append(f"- Video container: `{color_root}`")
    report_lines.append(f"- Label dirs (classes): **{len(label_dirs)}**")
    report_lines.append(f"- Clip samples: **{total_samples}**")
    report_lines.append("- Raw media format: frame folders (`*.jpg`), not encoded `.mp4` files")
    report_lines.append("")

    report_lines.append("## Annotation Files")
    if annotation_files:
        for p in annotation_files:
            report_lines.append(f"- `{p.relative_to(Path.cwd())}`")
    else:
        report_lines.append("- None detected")
    report_lines.append("")

    report_lines.append("## Split Files")
    if split_files:
        for p in split_files:
            report_lines.append(f"- `{p.relative_to(Path.cwd())}`")
    else:
        report_lines.append("- None detected")
    report_lines.append("")

    report_lines.append("## Sample Rows")
    report_lines.append("| video_path | label_id | frame_count |")
    report_lines.append("|---|---:|---:|")
    for row in sample_rows:
        report_lines.append(
            f"| `{row['video_path']}` | {row['label_id']} | {row['frame_count']} |"
        )
    report_lines.append("")

    out_path = Path(args.report_path)
    out_path.write_text("\n".join(report_lines), encoding="utf-8")

    print(f"[scan_csl] csl_root={csl_root}")
    print(f"[scan_csl] labels={len(label_dirs)}, samples={total_samples}")
    print(f"[scan_csl] annotation_files={len(annotation_files)}, split_files={len(split_files)}")
    print(f"[scan_csl] report written to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
