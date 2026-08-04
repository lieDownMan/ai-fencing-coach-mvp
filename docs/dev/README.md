# Development Overview

This page is the developer-facing overview for the current codebase.

For the repo landing page, use [../../README.md](../../README.md).
For the docs index, use [../README.md](../README.md).

## Read These First

1. [CURRENT_STATUS.md](CURRENT_STATUS.md)
2. [QUICKSTART.md](QUICKSTART.md)
3. [mvpspec.md](mvpspec.md)

Use this order on purpose:

- `CURRENT_STATUS.md` explains what the project is actually doing now
- `QUICKSTART.md` tells you how to run it
- `mvpspec.md` explains the higher-level product and research framing

## Current Runtime Entry Points

### `app.py`

Gradio clip-analysis UI.

Use this for:

- upload or record a clip
- annotated output video
- action table
- session history
- optional Gemini summary

### `realtime_app.py`

Local OpenCV webcam or local video source.

Use this for:

- live local webcam analysis
- on-screen warnings
- offline spoken voice cues through `pyttsx3`

### `web_realtime_app.py`

Browser-based live webcam streaming.

Use this for:

- browser webcam capture
- live analyzed frames in Gradio

## Main Code Paths

### Runtime

- `app.py`
- `realtime_app.py`
- `web_realtime_app.py`
- `database.py`
- `llm_agent.py`
- `realtime_voice_coach.py`
- `coach_playbook.json`

### Active inference package

- `inference/`

Important:

- The active runtime package is now `inference/`
- Older assumptions about `src/inference/` as the main runtime package are stale

### Still-used support code under `src/`

- `src/models/`
- `src/data/`
- `src/pose_estimation/`
- `src/preprocessing/`
- `src/tracking/`
- `src/training/`

## Documentation Roles

- [CURRENT_STATUS.md](CURRENT_STATUS.md): current reality and handoff
- [NEXT_AGENT_PROMPT.md](NEXT_AGENT_PROMPT.md): prompt for the next Codex agent
- [QUICKSTART.md](QUICKSTART.md): actual run commands
- [mvpspec.md](mvpspec.md): product and research spec
- [TRAINING.md](TRAINING.md): model-data workflow
- [CHECKPOINTS.md](CHECKPOINTS.md): model checkpoint expectations
- [CONTRIBUTING.md](CONTRIBUTING.md): coding conventions
- [HEURISTICS.md](HEURISTICS.md): posture heuristics calculations and logic details

## Current Priorities

The most valuable next work is:

1. local laptop runnability
2. webcam + voice-coaching reliability
3. handedness and target-tracking robustness
4. test-suite cleanup after the architecture shift

Not the first priority:

- new research writing
- major training refactors
- broad historical cleanup outside the active runtime path
