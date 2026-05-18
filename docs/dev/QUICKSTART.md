# Quick Start

This file is intentionally short. For the current runtime reality, read [CURRENT_STATUS.md](CURRENT_STATUS.md). For product scope and research framing, read [mvpspec.md](mvpspec.md).

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

The first real pose run needs YOLO pose weights. If the machine has network
access, Ultralytics can download `yolov8n-pose.pt` automatically. For offline
use, place `yolov8n-pose.pt` in the repo root or pass a local file with
`--pose-model`.

Optional Gemini setup:

```bash
printf 'GEMINI_API_KEY=your_key_here\n' > .env
```

If Gemini is not configured, clip analysis still produces a deterministic
`coach_playbook.json` summary listing every detected problem and how often it
appeared. If Gemini is configured, those same playbook details are included in
the LLM prompt. In the Analysis UI, use the `Use Gemini Summary` checkbox to
choose Gemini or the playbook-only summary for each run.

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
- generate a playbook summary, optionally polished by Gemini
- prioritize, mute, or limit which posture errors appear in the review

Use the `Processing` selector in the UI:

- `Balanced`: downscaled pose inference with good output quality
- `Fast`: lower pose resolution and smaller annotated output for quicker checks
- `Full Quality`: original-size pose inference and output

## Local Live Webcam Coaching

```bash
python -m src.realtime.realtime_app --source 0 --mode "Free Bouting" --target-side left
```

Other modes:

```bash
python -m src.realtime.realtime_app --source 0 --mode "Footwork"
python -m src.realtime.realtime_app --source 0 --mode "Target Practice"
```

Useful local flags:

```bash
python -m src.realtime.realtime_app --source 0 --target-side right
python -m src.realtime.realtime_app --source 0 --pose-model yolov8n-pose.pt
python -m src.realtime.realtime_app --source 0 --no-voice
python -m src.realtime.realtime_app --source 0 --pose-backend mock --no-voice
python -m src.realtime.realtime_app --source 0 --focus-errors stance_too_high,bounce_excessive
python -m src.realtime.realtime_app --source 0 --mute-errors guard_dropped
python -m src.realtime.realtime_app --source 0 --only-errors stance_too_high
```

Use a local video file instead of a webcam:

```bash
python -m src.realtime.realtime_app --source path/to/video.mp4 --mode "Free Bouting"
```

This is the best option for:

- local laptop webcam
- local audio playback
- direct real-time spoken cues
- ranked feedback where voice speaks one selected cue and the HUD shows the top issues
- optional focus/mute/only filters for the errors you want to train today

Target selection now locks onto the first usable detection, not only frame 0.
If ByteTrack's optional `lap` dependency is missing, the runtime falls back to
plain pose detection without persistent track IDs.

## Browser Live Webcam Streaming

```bash
python web_realtime.py
```

Open:

- `http://127.0.0.1:8000`

Use this when you want:

- browser webcam capture
- streaming analyzed frames
- a web-based live demo
- realtime heuristic metrics in a browser debug panel
- browser controls for focused, muted, or only-selected feedback errors

The page includes a heuristic debug panel. You can enable/disable it, choose a
heuristic, tune the rolling window size, and reset the target lock from the
browser. It also includes feedback focus controls, so you can prioritize
specific errors, mute noisy errors, or only show/speak focused errors. The
camera source is still opened by the backend; for phone-camera setups such as
DroidCam, put the DroidCam video URL in `Camera Source`.

## Heuristic Visualizer UI

```bash
python heuristic_visualizer.py
```

Open:

- `http://127.0.0.1:7862` or the printed port if `7862` is busy

Use this when you want:

- see the target skeleton overlaid on a clip frame by frame
- inspect one heuristic at a time, or all heuristics together
- see the metric value and threshold used by each heuristic
- find the exact timestamp where the current runtime emitted a posture alert
- download a CSV log and highlighted HTML log under `web_outputs/heuristic_debug/logs/`

The default `Debug Granularity` is `Frame by Frame`. It computes heuristic
metrics from a rolling skeleton window ending at each frame. Use `Action
Segment` only when you want the older segment-level summary.

## Realtime Heuristic Visualizer

```bash
python realtime_heuristic_visualizer.py --source 0 --target-side left --mode "Footwork" --heuristic all
```

Use this when you want:

- live webcam skeleton overlay
- live target-lock debugging without voice cues
- per-frame rolling heuristic values before tuning thresholds

This is the lower-latency local OpenCV version of the debug panel in
`web_realtime.py`.

Useful flags:

```bash
python realtime_heuristic_visualizer.py --source 0 --heuristic stance_too_high
python realtime_heuristic_visualizer.py --source path/to/video.mp4 --log-csv web_outputs/heuristic_debug/logs/realtime_debug.csv
python realtime_heuristic_visualizer.py --source 0 --pose-backend mock
```

Controls:

- `q`: quit
- `r`: reset target lock

## Notes About Local vs Workstation Usage

- For local webcam plus local voice, prefer `python -m src.realtime.realtime_app`.
- For workstation compute plus browser webcam, use `web_realtime.py`.
- Voice coaching uses `pyttsx3`, so audio comes out on the machine running the backend.
- Gradio apps are local by default. Set `GRADIO_SHARE=1` before launch only when you need a public Gradio share link.

## Prepare Training Data

Prepare the public FFD dataset. The script supports both the older Kinect
`*_Body.mat` layout and the local vision-only folder layout under `FFD/FFD`:

```bash
python scripts/prepare_ffd.py --ffd-root FFD/FFD --output data/training/ffd_prepared.npz --pose-backend ultralytics --pose-model yolov8n-pose.pt
```

Quick structural smoke check without real pose extraction:

```bash
python scripts/prepare_ffd.py --ffd-root FFD/FFD --output data/training/ffd_mock_smoke.npz --pose-backend mock
```

For the local vision-only FFD, raw `.mov` recordings and `6_Idle` clips are
ignored because the current FenceNet head has six classes: `R`, `IS`, `WW`,
`JS`, `SF`, and `SB`.

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
- the OpenCV live window from `python -m src.realtime.realtime_app`
- the OpenCV debug window from `realtime_heuristic_visualizer.py`
- the browser stream from `web_realtime.py`
- `web_outputs/heuristic_debug/` for heuristic visualizer uploads and overlays
- `fencing_coach.db` for persisted users and sessions
- `reports/` for saved debug JSON or prior artifacts
- `web_outputs/` for browser-generated files when applicable

## If Something Fails

- `ModuleNotFoundError: cv2`: install dependencies with `pip install -r requirements.txt`.
- `Ultralytics is not installed`: install dependencies and ensure YOLO pose weights are available.
- `google-genai module not installed`: reinstall requirements or check your venv.
- `GEMINI_API_KEY not set`: create `.env` in the repo root and restart `app.py`.
- `pyttsx3` voice does not play: verify local OS audio support and package installation.
- `ModuleNotFoundError: lap`: reinstall requirements. The live app can continue without persistent track IDs, but ByteTrack is more stable with `lap` installed.
- webcam unavailable in `python -m src.realtime.realtime_app`: confirm the camera is attached to the same machine.
- browser webcam works but no audio: remember `pyttsx3` speaks on the backend machine, not the browser client.
