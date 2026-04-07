# Quick Start

This file is intentionally short. For product scope and workflow, read [mvpspec.md](mvpspec.md).

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run Interactive Mode

```bash
python app.py --interactive
```

## Process A Video

```bash
python app.py --video path/to/bout.mp4 --fencer-id athlete_001 --device auto --pose-backend mock
```

Use the local ignored sample video if it is present:

```bash
python app.py --video video/fencing_match.mp4 --fencer-id athlete_001 --device cpu --pose-backend mock
```

With an athlete name:

```bash
python app.py --video path/to/bout.mp4 --fencer-id athlete_001 --fencer-name "John Smith" --pose-backend mock
```

Use CUDA if available:

```bash
python app.py --video path/to/bout.mp4 --fencer-id athlete_001 --device cuda --pose-backend mock
```

Use BiFenceNet:

```bash
python app.py --video path/to/bout.mp4 --fencer-id athlete_001 --use-bifencenet --pose-backend mock
```

Load a model checkpoint:

```bash
python app.py --video path/to/bout.mp4 --fencer-id athlete_001 --model weights/fencenet/model_best.pth --pose-backend mock
```

Use a real YOLO pose backend when `ultralytics` and the pose model are installed:

```bash
python app.py --video path/to/bout.mp4 --fencer-id athlete_001 --pose-backend ultralytics --pose-model yolov8n-pose.pt
```

Write a JSON report for review or downstream analysis:

```bash
python app.py --video path/to/bout.mp4 --fencer-id athlete_001 --pose-backend mock --report reports/bout_report.json
```

Write an annotated review video with fencer boxes and a too-close distance cue:

```bash
python app.py --video video/fencing_match.mp4 --fencer-id athlete_001 --device cpu --pose-backend mock --annotated-video video/fencing_match_processed.mp4
```

If `output.save_reports: true` is enabled in `config.yaml`, the CLI writes an auto-named report under `output.reports_dir` unless `--no-report` is passed.

## Help

```bash
python app.py --help
```

## Outputs To Check

- `data/fencer_profiles/` for athlete profile JSON files.
- `reports/` for JSON video reports when enabled by config or `--report`.
- `video/fencing_match_processed.mp4` or your `--annotated-video` path for annotated video output.
- CLI summary output for frames processed and post-bout feedback.
- The OpenCV preview window for interactive visual debugging.

## If Something Fails

- `ModuleNotFoundError: cv2`: install dependencies with `pip install -r requirements.txt`.
- `Ultralytics is not installed`: use `--pose-backend mock` or install dependencies before using `--pose-backend ultralytics`.
- CUDA unavailable: use `--device cpu` or install CUDA-compatible PyTorch.
- `Video file not found`: verify the file path. The CLI exits with status `1`.
- Low-quality or repeated action labels: provide trained model weights. Randomly initialized weights are only useful for pipeline smoke tests.
- Low FPS: use CPU/GPU settings appropriate to the machine and reduce video resolution if needed.
