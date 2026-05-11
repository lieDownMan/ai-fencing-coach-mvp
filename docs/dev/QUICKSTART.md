# Quick Start

This file is intentionally short. For the current runtime reality, read [CURRENT_STATUS.md](CURRENT_STATUS.md). For product scope and research framing, read [mvpspec.md](mvpspec.md).

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Optional Gemini setup:

```bash
printf 'GEMINI_API_KEY=your_key_here\n' > .env
```

## Clip Analysis UI

```bash
python app.py
```

Open:

- `http://127.0.0.1:7860`

Use this when you want:

- upload or record a clip
- save session history
- generate annotated output
- optionally generate a Gemini summary

## Local Live Webcam Coaching

```bash
python realtime_app.py --source 0 --mode "Free Bouting"
```

Other modes:

```bash
python realtime_app.py --source 0 --mode "Footwork"
python realtime_app.py --source 0 --mode "Target Practice"
```

Use a local video file instead of a webcam:

```bash
python realtime_app.py --source path/to/video.mp4 --mode "Free Bouting"
```

This is the best option for:

- local laptop webcam
- local audio playback
- direct real-time spoken cues

## Browser Live Webcam Streaming

```bash
python web_realtime_app.py
```

Open:

- `http://127.0.0.1:7861`

Use this when you want:

- browser webcam capture
- streaming analyzed frames
- a web-based live demo

## Notes About Local vs Workstation Usage

- For local webcam plus local voice, prefer `realtime_app.py`.
- For workstation compute plus browser webcam, use `web_realtime_app.py`.
- Voice coaching uses `pyttsx3`, so audio comes out on the machine running the backend.

## Prepare Training Data

Prepare the public FFD dataset:

```bash
python scripts/prepare_ffd.py --ffd-root /path/to/ffd --output data/training/ffd_prepared.npz
```

Write a starter custom-label CSV:

```bash
python scripts/prepare_labeled_clips.py --write-template labels/clip_labels_template.csv
```

Prepare labeled clips from your own videos:

```bash
python scripts/prepare_labeled_clips.py --labels-csv labels/my_clips.csv --output data/training/my_labeled_clips.npz --pose-backend ultralytics --pose-model yolov8n-pose.pt
```

Train a model:

```bash
python -m src.training.train_fencenet --dataset data/training/ffd_prepared.npz --output-dir weights/fencenet_ffd_run1
```

## Outputs To Check

- `annotated_output.mp4` from `app.py`
- the OpenCV live window from `realtime_app.py`
- the browser stream from `web_realtime_app.py`
- `fencing_coach.db` for persisted users and sessions
- `reports/` for saved debug JSON or prior artifacts
- `web_outputs/` for browser-generated files when applicable

## If Something Fails

- `ModuleNotFoundError: cv2`: install dependencies with `pip install -r requirements.txt`.
- `Ultralytics is not installed`: install dependencies and ensure YOLO pose weights are available.
- `google-genai module not installed`: reinstall requirements or check your venv.
- `GEMINI_API_KEY not set`: create `.env` in the repo root and restart `app.py`.
- `pyttsx3` voice does not play: verify local OS audio support and package installation.
- webcam unavailable in `realtime_app.py`: confirm the camera is attached to the same machine.
- browser webcam works but no audio: remember `pyttsx3` speaks on the backend machine, not the browser client.
