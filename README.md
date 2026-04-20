# AI Fencing Coach MVP

This repository is a research-oriented MVP for AI-assisted fencing coaching. It focuses on ordinary-video analysis, fencing-specific movement understanding, and feedback workflows for beginner and intermediate fencers.

The current project direction is:

- a coaching support tool, not a referee replacement
- a commodity-camera pipeline, not a specialized lab setup
- an HCI artifact about feedback usefulness, not only model accuracy

## Quick Start

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app.py --interactive
```

Process a video:

```bash
python app.py --video path/to/bout.mp4 --fencer-id athlete_001 --device auto --pose-backend mock
```

Train a baseline model:

```bash
python train.py --dataset data/training/ffd_prepared.npz --output-dir weights/fencenet_ffd_run1 --model-type fencenet
```

## Documentation

The full project documentation now lives under [`docs/`](docs).

Start here:

- [Full Project Overview](docs/README.md)
- [MVP Spec](docs/mvpspec.md)
- [Quickstart](docs/QUICKSTART.md)
- [Training Guide](docs/TRAINING.md)
- [Checkpoints Guide](docs/CHECKPOINTS.md)

Research workflow documents:

- [Related Work Memo](docs/RELATED_WORK.md)
- [Research Brief](docs/RESEARCH_BRIEF.md)
- [Evidence Workflow](docs/EVIDENCE_WORKFLOW.md)
- [Interview Guide](docs/INTERVIEW_GUIDE.md)
- [Chinese Interview Script](docs/INTERVIEW_SCRIPT_ZH.md)

## Current Scope

Implemented in the current code path:

- side-view fencing video input
- pose extraction with `mock` and `ultralytics`
- two-fencer tracking for visualization
- paper-aligned FenceNet / BiFenceNet architecture support
- sliding-window footwork recognition
- annotated video output
- local browser demo
- analytical coaching fallback
- profile storage and JSON report output

Still planned:

- trained production-quality fencing checkpoints
- stronger identity persistence through crossings and occlusion
- coach-validated live feedback heuristics
- real LLM integration

## Project Structure

```text
ai-fencing-coach-mvp/
├── README.md
├── app.py
├── web_app.py
├── train.py
├── config.yaml
├── requirements.txt
├── docs/
├── scripts/
├── src/
├── tests/
├── data/
└── video/
```
