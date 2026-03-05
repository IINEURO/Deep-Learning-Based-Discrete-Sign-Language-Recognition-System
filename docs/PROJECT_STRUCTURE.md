# Project Structure

## Core Runtime

- `src/slr_baseline/`: model/data/features/keypoints core code
- `train.py`: training entry
- `infer.py`: offline inference entry
- `gradio_app.py`: UI entry
- `realtime_demo.py`: camera demo entry

## Data Pipeline Scripts

- `scripts/scan_csl.py`: scan dataset and generate `docs/report.md`
- `scripts/prepare_meta.py`: build `meta/vocab_gloss.json` + `meta/manifest.csv`
- `scripts/extract_keypoints.py`: extract keypoints into `dataset_processed/`
- `scripts/visualize_samples.py`: render sample videos into `outputs/debug_vis/`

## Data and Artifacts

- `dataset/`: raw dataset
- `dataset_processed/`: primary processed dataset (active)
- `meta/`: manifest + vocabulary files
- `checkpoints/`: primary model checkpoints (active)
- `outputs/`: runtime outputs and visualizations
- `experiments/`: archived historical experiments (old checkpoints and processed sets)

## Documents and Assets

- `docs/`: manuals and reports
- `assets/`: logo and UI assets
