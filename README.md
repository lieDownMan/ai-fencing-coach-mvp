# AI Fencing Coach MVP

This repository is a desktop MVP for an AI-assisted fencing coach. It uses video input, pose estimation, fencing-specific motion analysis, and coaching feedback to help beginner and intermediate fencers review distance, footwork, and tactical tendencies.

The current positioning is: a coaching support tool, not a referee replacement. The product value should be judged by whether real fencers and coaches can use the feedback during practice or post-bout review, not by model accuracy alone.

## Current Scope

Implemented and tested in the current code path:

- Side-view fencing video from a webcam or imported video file.
- Pose extraction backends: `mock` for deterministic development/tests and `ultralytics` for YOLO pose when the dependency and weights are installed.
- Single selected fencer skeleton per frame, using the largest detected person for the Ultralytics backend.
- Explicit 10-joint, 20-channel skeleton feature order for FenceNet/BiFenceNet inference.
- Sliding-window FenceNet/BiFenceNet-style six-class footwork recognition.
- Pattern analysis for action frequency, offensive/defensive ratio, JS/SF ratio, repeated patterns, and average confidence.
- Athlete profile storage for longitudinal review.
- LLM coaching interface with deterministic analytical fallback. A real LLM backend is not loaded by default in this MVP.
- Metadata-aware FenceNet/BiFenceNet checkpoint loading with CLI status for checkpoint vs random weights.
- Optional JSON report output for processed videos, including classification windows, action statistics, runtime metadata, and feedback.
- CLI/config workflow and OpenCV dashboard rendering, including headless-safe UI tests.

Still planned or research-facing:

- Robust two-fencer tracking and left/right fencer assignment.
- Distance, stance-width, and engagement heuristics for live form feedback.
- Trained fencing model checkpoints. The loader and expected format are documented, but trained weights are not included yet.
- Real LLM model loading or API integration.
- Real Ultralytics pose smoke testing in this environment. The current venv does not have `ultralytics` installed.

Out of scope for this MVP:

- Full electronic refereeing or official scoring.
- Blade tracking and fine-grained weapon contact analysis.
- Multi-camera 3D reconstruction.
- Mobile or web deployment.
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
     -> current path: one selected fencer skeleton per frame
     -> planned path: two-fencer tracking and left/right assignment
  -> skeleton normalization into a 10-joint / 20-channel feature tensor
  -> sliding-window FenceNet/BiFenceNet six-class action recognition
  -> pattern analysis and athlete profile update
  -> coaching feedback
     -> current path: analytical fallback from tracked stats
     -> planned path: real LLM-generated coaching where appropriate
  -> CLI summary and/or OpenCV dashboard
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

Write a JSON report:

```bash
python app.py --video path/to/bout.mp4 --fencer-id athlete_001 --pose-backend mock --report reports/bout_report.json
```

Run the local ignored sample video if present:

```bash
python app.py --video video/fencing_match.mp4 --fencer-id athlete_001 --device cpu --pose-backend mock
```

See [QUICKSTART.md](QUICKSTART.md) for the minimal command reference.

## Project Structure

```text
ai-fencing-coach-mvp/
├── app.py
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
