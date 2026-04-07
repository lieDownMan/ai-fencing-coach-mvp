# AI Fencing Coach MVP

This repository is a desktop MVP for an AI-assisted fencing coach. It uses video input, pose estimation, fencing-specific motion analysis, and coaching feedback to help beginner and intermediate fencers review distance, footwork, and tactical tendencies.

The current positioning is: a coaching support tool, not a referee replacement. The product value should be judged by whether real fencers and coaches can use the feedback during practice or post-bout review, not by model accuracy alone.

## Current Scope

Implemented and tested in the current code path:

- Side-view fencing video from a webcam or imported video file.
- Pose extraction backends: `mock` for deterministic development/tests and `ultralytics` for YOLO pose when the dependency and weights are installed.
- Opt-in real Ultralytics pose smoke coverage for `video/fencing_match.mp4` when `ultralytics` and a local YOLO pose model are available.
- Side-based two-fencer candidate tracking for visualization: keep the two largest pose candidates per frame and label them `fencer_L`/`fencer_R` by horizontal position.
- Prototype distance feedback: mark global `too_close` frames from average tracked height and show per-fencer too-close status against each fencer's own detected height.
- Annotated MP4 output with fencer boxes, skeleton keypoints, engagement-distance line, dual left/right HUD panels, speed/movement cues, global action label, optional height calibration, optional web-friendly downscaling plus H.264 transcoding, and too-close warning banner.
- Local no-dependency browser demo at `web_app.py` for processing a video, reviewing the annotated MP4, and reading summary metrics without typing the full CLI command.
- Single selected fencer skeleton remains the classifier input, using the largest detected person so existing FenceNet/BiFenceNet inference stays stable.
- Explicit 10-joint, 20-channel skeleton feature order for FenceNet/BiFenceNet inference.
- Sliding-window FenceNet/BiFenceNet-style six-class footwork recognition.
- Pattern analysis for action frequency, offensive/defensive ratio, JS/SF ratio, repeated patterns, and average confidence.
- Athlete profile storage for longitudinal review.
- LLM coaching interface with deterministic analytical fallback. A real LLM backend is not loaded by default in this MVP.
- Metadata-aware FenceNet/BiFenceNet checkpoint loading with CLI status for checkpoint vs random weights.
- Optional JSON report output for processed videos, including classification windows, action statistics, runtime metadata, and feedback.
- CLI/config workflow and OpenCV dashboard rendering, including headless-safe UI tests.

Still planned or research-facing:

- Robust identity persistence through fencer crossing, occlusion, and exchange resets.
- Coach-validated distance thresholds, stance-width checks, limb/reach calibration, and recovery/timing heuristics for live form feedback.
- Trained fencing model checkpoints. The loader and expected format are documented, but trained weights are not included yet.
- Real LLM model loading or API integration.

Out of scope for this MVP:

- Full electronic refereeing or official scoring.
- Blade tracking and fine-grained weapon contact analysis.
- Multi-camera 3D reconstruction.
- Hosted production deployment or mobile app distribution.
- Medical, professional safety, or certification claims.

## Documentation Map

Use these documents as the current source of truth:

| Document | Purpose |
| --- | --- |
| [mvpspec.md](mvpspec.md) | Canonical workflow, scope, and research positioning spec. Start here for design decisions. |
| [QUICKSTART.md](QUICKSTART.md) | Minimal setup and run commands. |
| [CHECKPOINTS.md](CHECKPOINTS.md) | Expected FenceNet/BiFenceNet checkpoint format and loading behavior. |
| [README_zh.md](README_zh.md) | Short Chinese summary and doc navigation. |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Development and contribution conventions. |

These files are retained as pointers for continuity:

| Document | Status |
| --- | --- |
| [detailedstructure.md](detailedstructure.md) | Superseded by [mvpspec.md](mvpspec.md). |
| [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) | Historical generated implementation snapshot. |

## System Workflow

The canonical workflow is:

```text
Video input
  -> pose estimation
     -> current path: side-based two-fencer candidates for visualization plus one selected skeleton for classification
     -> planned path: robust identity persistence through crossings and occlusion
  -> skeleton normalization into a 10-joint / 20-channel feature tensor
  -> sliding-window FenceNet/BiFenceNet six-class action recognition
  -> pattern analysis and athlete profile update
  -> coaching feedback
     -> current path: analytical fallback from tracked stats
     -> planned path: real LLM-generated coaching where appropriate
  -> CLI summary, local browser demo, and/or OpenCV dashboard
```

The key product question is not "Can we classify fencing movement?" The stronger HCI question is "Can a fencer or coach change practice behavior because the system gives timely, understandable, trustworthy feedback?"

## Target Actions

FenceNet-style classification uses six target footwork classes:

| Code | Action | Category |
| --- | --- | --- |
| `R` | Rapid lunge | Offensive |
| `IS` | Incremental speed lunge | Offensive |
| `WW` | With waiting lunge | Offensive |
| `JS` | Jumping sliding lunge | Offensive |
| `SF` | Step forward | Neutral |
| `SB` | Step backward | Defensive |

## Quick Start

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app.py --interactive
```

Process a video file:

```bash
python app.py --video path/to/bout.mp4 --fencer-id athlete_001 --device auto --pose-backend mock
```

Write a JSON report with classification windows and two-fencer tracking frames:

```bash
python app.py --video path/to/bout.mp4 --fencer-id athlete_001 --pose-backend mock --report reports/bout_report.json
```

Write an annotated video for visual review:

```bash
python app.py --video video/fencing_match.mp4 --fencer-id athlete_001 --device cpu --pose-backend mock --left-height-cm 170 --right-height-cm 185 --annotated-max-width 1280 --annotated-video video/fencing_match_processed.mp4
```

The height flags are optional; without them, the HUD uses detected bounding-box height only. Limb length and reach calibration are intentionally future work.

Run the local ignored sample video if present:

```bash
python app.py --video video/fencing_match.mp4 --fencer-id athlete_001 --device cpu --pose-backend mock
```


Run the local browser demo:

```bash
python web_app.py
```

Open `http://127.0.0.1:7860`, choose the sample video or another server-side video path, set left/right heights, and click **Process Video**. Use `ultralytics` for real CV boxes; `mock` is only for deterministic pipeline checks.

If port `7860` is already in use, start the demo on another port:

```bash
python web_app.py --port 7861
```

`Annotated max width` is the maximum width, in pixels, of the exported annotated MP4. For example, `1280` downscales a `1908x920` output to about `1280x616` for smoother browser playback. It does not change pose detection, tracking, or model inference.

See [QUICKSTART.md](QUICKSTART.md) for the minimal command reference.

## Project Structure

```text
ai-fencing-coach-mvp/
├── app.py
├── web_app.py
├── config.yaml
├── requirements.txt
├── mvpspec.md
├── QUICKSTART.md
├── README.md
├── README_zh.md
├── CONTRIBUTING.md
├── data/
├── video/
├── src/
│   ├── pose_estimation/
│   ├── preprocessing/
│   ├── models/
│   ├── tracking/
│   ├── llm_agent/
│   └── app_interface/
└── tests/
```

## Research Positioning Questions

Before treating this as a publishable HCI project, the next step is to answer these with evidence:

- Who specifically struggles with current fencing feedback, and what workaround do they use now?
- What have we directly observed in practice sessions, lessons, or bout reviews?
- Which gap are we claiming: a new capability, a better coaching experience, or a lower-cost version of an existing workflow?
- What changed recently that makes this feasible now: pose models, local LLMs, commodity cameras, or coaching constraints?
- Can we test with real beginner/intermediate fencers and at least one coach within the project timeline?

Those answers should drive the next research brief and the next implementation milestone.

## License

This project is currently documented for educational and HCI research use. Add a repository `LICENSE` file before making a public release claim.
