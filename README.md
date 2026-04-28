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
unzip FFD.zip
python convert_to_json.py
python train_fencenet.py --data_dir data/json_samples
```

## FFD Dataset Note

- Use the local `FFD.zip` bundle as the source of truth when preparing training data.
- The lightweight `FFD/` directory in git is not a complete substitute for the local archive.
- `convert_to_json.py` converts the extracted clips into the JSON format expected by `train_fencenet.py`.
- The converter now accepts both `3_IS/` and `3_IR/` folder names as the `IS` class, because the local archive uses `3_IR/`.
- The converter also canonicalizes `front_*` joints from the fencer's screen side, so retraining does not hard-code the right arm as the weapon arm.

## Documentation

The full project documentation now lives under [`docs/`](docs).

Start here:

- [Docs Index](docs/README.md)
- [Development Overview](docs/dev/README.md)
- [Research Index](docs/research/README.md)

### Docs Map

Development docs:

- [Project Overview](docs/dev/README.md)
- [MVP Spec](docs/dev/mvpspec.md)
- [Quickstart](docs/dev/QUICKSTART.md)
- [Training Guide](docs/dev/TRAINING.md)
- [Checkpoints Guide](docs/dev/CHECKPOINTS.md)
- [Contributing Guide](docs/dev/CONTRIBUTING.md)
- [Chinese Overview](docs/dev/README_zh.md)

Research workflow documents:

- [Research Brief](docs/research/RESEARCH_BRIEF.md)
- [Related Work Memo](docs/research/RELATED_WORK.md)
- [Evidence Workflow](docs/research/EVIDENCE_WORKFLOW.md)
- [Interview Guide](docs/research/INTERVIEW_GUIDE.md)
- [Interview Record Template](docs/research/INTERVIEW_RECORD_TEMPLATE.md)
- [Interview Synthesis Template](docs/research/INTERVIEW_SYNTHESIS_TEMPLATE.md)
- [Research Brief Revision Checklist](docs/research/RESEARCH_BRIEF_REVISION_CHECKLIST.md)
- [Chinese Interview Script](docs/research/INTERVIEW_SCRIPT_ZH.md)
- [Chinese Proposal Intro](docs/research/PROPOSAL_INTRO_ZH.md)

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
- target re-locking when YOLO/ByteTrack changes track IDs mid-video
- sparse-frame-safe annotation for gatekeeper-skipped frames
- front-limb canonicalization driven by the selected target side, so the pipeline uses the actual leading arm / leg instead of assuming right-handed poses
- optional `weapon_hand` override (`auto`, `left`, `right`) for cases where screen-side inference is not enough

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
├── gradio_app.py
├── convert_to_json.py
├── train_fencenet.py
├── config.yaml
├── requirements.txt
├── FFD/
├── docs/
│   ├── README.md
│   ├── dev/
│   └── research/
├── scripts/
├── src/
├── tests/
├── data/
└── video/
```
