# AI Fencing Coach MVP

This repository is an AI-assisted fencing coaching prototype focused on:

- clip analysis through a Gradio UI
- live local webcam coaching
- browser webcam streaming
- rule-based biomechanics feedback
- optional Gemini session summaries

The project is a coaching aid, not a referee replacement.

## Start Here

- [docs/dev/CURRENT_STATUS.md](docs/dev/CURRENT_STATUS.md): best handoff and reality-check document
- [docs/dev/QUICKSTART.md](docs/dev/QUICKSTART.md): actual run commands
- [docs/dev/README.md](docs/dev/README.md): development overview
- [docs/README.md](docs/README.md): full docs index

## Main Runtime Modes

### 1. Clip Analysis UI

```bash
python app.py
```

Use this when you want:

- upload or record a clip
- annotated output video
- action table
- session history
- optional Gemini summary

### 2. Local Live Webcam Coaching

```bash
python realtime_app.py --source 0 --mode "Free Bouting" --target-side left
```

Use this when you want:

- direct webcam on the same machine
- local OpenCV window
- immediate spoken coaching cues through `pyttsx3`

This is the best choice for a **local laptop demo**.

Useful live flags:

```bash
python realtime_app.py --source 0 --target-side right
python realtime_app.py --source 0 --pose-model yolov8n-pose.pt
python realtime_app.py --source 0 --no-voice
```

### 3. Browser Live Webcam Streaming

```bash
python web_realtime_app.py
```

Use this when you want:

- browser webcam streaming
- live analyzed frames in a web page

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Optional Gemini setup:

```env
GEMINI_API_KEY=your_key_here
```

Put that in a local `.env` file in the repo root before starting `app.py`.

For real pose inference, keep `yolov8n-pose.pt` in the repo root or let
Ultralytics download it on first run. Persistent webcam target IDs use the
optional `lap` package listed in `requirements.txt`; the live app falls back to
non-persistent pose detections if `lap` is missing.

## Current Architecture

High-level runtime flow:

```text
camera or video
  -> YOLO pose extraction
  -> target tracking + activity gatekeeper
  -> sliding-window FenceNet inference
  -> heuristics engine
  -> video overlay / action table / voice cue / optional LLM summary
```

Important runtime files:

- `app.py`
- `realtime_app.py`
- `web_realtime_app.py`
- `realtime_voice_coach.py`
- `coach_playbook.json`
- `inference/`
- `llm_agent.py`
- `database.py`

## Important Notes

- The active inference package is now top-level `inference/`, not `src/inference/`.
- Voice cues do not require Gemini. They use offline `pyttsx3`.
- Gradio public share links are off by default for local runs. Set `GRADIO_SHARE=1` only when you need a public tunnel.
- The docs were cleaned to match the current code, but older branch history may still mention removed flows such as `web_app.py` or old CLI-only usage.
- For the most accurate current state, trust [docs/dev/CURRENT_STATUS.md](docs/dev/CURRENT_STATUS.md) over older historical notes.
