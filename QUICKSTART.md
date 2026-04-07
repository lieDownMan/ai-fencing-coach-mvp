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

Write an annotated review video with dual fencer HUDs, current global action, speed/movement cues, and a too-close distance cue:

```bash
python app.py --video video/fencing_match.mp4 --fencer-id athlete_001 --device cpu --pose-backend mock --left-height-cm 170 --right-height-cm 185 --annotated-max-width 1280 --annotated-video video/fencing_match_processed.mp4
```

The height flags are optional. Limb length and reach calibration are future work, not current CLI inputs.


## Browser Demo

```bash
python web_app.py
```

Open `http://127.0.0.1:7860`, keep `ultralytics` selected for real CV boxes, and click **Process Video**. The browser demo writes generated MP4s/reports under `web_outputs/`.

If port `7860` is already in use, run `python web_app.py --port 7861` and open `http://127.0.0.1:7861` instead.

`Annotated max width` only downscales the exported annotated MP4 for browser playback. It does not change pose detection, tracking, or inference.

If `output.save_reports: true` is enabled in `config.yaml`, the CLI writes an auto-named report under `output.reports_dir` unless `--no-report` is passed.

## Help

```bash
python app.py --help
```

## Outputs To Check

- `data/fencer_profiles/` for athlete profile JSON files.
- `reports/` for JSON video reports when enabled by config or `--report`.
- `video/fencing_match_processed.mp4` or your `--annotated-video` path for annotated video output.
- `web_outputs/` for browser demo MP4s, reports, and temporary profiles.
- CLI summary output for frames processed and post-bout feedback.
- The OpenCV preview window for interactive visual debugging.

## If Something Fails

- `ModuleNotFoundError: cv2`: install dependencies with `pip install -r requirements.txt`.
- `Ultralytics is not installed`: use `--pose-backend mock` or install dependencies before using `--pose-backend ultralytics`.
- CUDA unavailable: use `--device cpu` or install CUDA-compatible PyTorch.
- `Video file not found`: verify the file path. The CLI exits with status `1`.
- Low-quality or repeated action labels: provide trained model weights. Randomly initialized weights are only useful for pipeline smoke tests.
- Low FPS: use CPU/GPU settings appropriate to the machine and reduce video resolution if needed.
