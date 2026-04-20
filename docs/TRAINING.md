# Training Guide

This repository now includes a first-pass training workflow for FenceNet and BiFenceNet.

## 1. Install Dependencies

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 2. Prepare FFD

Download and unpack the public Fencing Footwork Dataset (FFD), then convert it into a model-ready `.npz` bundle:

```bash
python scripts/prepare_ffd.py \
  --ffd-root /path/to/ffd \
  --output data/training/ffd_prepared.npz \
  --summary-json data/training/ffd_summary.json

`/path/to/ffd` is a placeholder in the example above. Replace it with the real directory where you unpacked FFD.
```

Notes:

- The public FFD dataset already has action labels.
- FFD uses Kinect body data, so this script maps Kinect `head` to the model's `nose` normalization anchor.
- The conversion follows the paper-style preprocessing: 28-frame windows, up to 10 windows per sequence, random starts limited to the first 20 frames.

## 3. Train A Baseline

Train FenceNet:

```bash
python train.py \
  --dataset data/training/ffd_prepared.npz \
  --output-dir weights/fencenet_ffd_run1 \
  --model-type fencenet \
  --epochs 20 \
  --batch-size 32
```

Train BiFenceNet:

```bash
python train.py \
  --dataset data/training/ffd_prepared.npz \
  --output-dir weights/bifencenet_ffd_run1 \
  --model-type bifencenet \
  --epochs 20 \
  --batch-size 32
```

Hold out one subject for validation:

```bash
python train.py \
  --dataset data/training/ffd_prepared.npz \
  --output-dir weights/fencenet_holdout_fencer01 \
  --model-type fencenet \
  --holdout-subject fencer_01
```

Each training run writes:

- `best.pt`
- `last.pt`
- `metrics.json`

The checkpoints are compatible with `app.py --model ...`.

## 4. Load A Trained Checkpoint In The App

```bash
python app.py \
  --video path/to/bout.mp4 \
  --fencer-id athlete_001 \
  --model-type fencenet \
  --model weights/fencenet_ffd_run1/best.pt \
  --pose-backend ultralytics
```

## 5. Prepare Your Own Labeled Clips

Write a starter CSV template:

```bash
python scripts/prepare_labeled_clips.py --write-template labels/clip_labels_template.csv
```

CSV columns:

- `video_path` required
- `label` required
- `start_frame` optional, defaults to `0`
- `end_frame` optional, defaults to the end of the video
- `subject_id` optional
- `sample_id` optional
- `notes` optional

Example row:

```csv
video_path,label,start_frame,end_frame,subject_id,sample_id,notes
video/my_session_clip_001.mp4,SF,0,42,athlete_001,clip_001,single clean step-forward repetition
```

Convert labeled clips into a prepared dataset:

```bash
python scripts/prepare_labeled_clips.py \
  --labels-csv labels/my_clips.csv \
  --output data/training/my_labeled_clips.npz \
  --pose-backend ultralytics \
  --pose-model yolov8n-pose.pt
```

Notes:

- This path extracts skeletons with the existing pose pipeline.
- It then normalizes each labeled clip and resamples it to 28 frames.
- Use `--pose-backend mock` only for pipeline checks, not for real training data.

## 6. Recommended Order

1. Train a baseline on FFD.
2. Run that checkpoint through the app.
3. Label a small set of your own clips.
4. Fine-tune or retrain with the custom prepared dataset.
