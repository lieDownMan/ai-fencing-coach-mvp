# Current Status and Handoff

Status: Active handoff note
Last updated: 2026-05-11
Purpose: Give the next developer or agent an accurate picture of the current codebase, what changed recently, what is actually runnable, and what still needs attention.

## 1. Current Branch and Direction

- Current branch: `main`
- Latest reviewed remote commit: `a755b8a`
- Recent headline change: `add realtime module`

Current team direction:

- move from workstation-centric usage toward **local laptop usage**
- keep the new real-time modules
- make the docs match the current codebase
- hand off cleanly to a new Codex agent running locally

## 2. What the Project Is Right Now

This project is currently an AI-assisted fencing coaching prototype with:

- pose extraction from YOLO pose
- sliding-window FenceNet action recognition
- rule-based biomechanics and coaching heuristics
- Gradio-based clip analysis UI
- optional Gemini post-session summary
- new offline voice-coaching for immediate spoken cues
- new real-time webcam and browser-streaming entrypoints

The system is no longer just a batch video analysis tool. It now has three distinct runtime modes.

## 3. The Three Main Entry Points

### A. `app.py`

Use this for:

- upload or record a clip
- run full analysis
- save user/session history
- generate annotated video
- generate action table
- generate a playbook summary, optionally polished by Gemini

Key characteristics:

- Gradio UI
- uses `Database`
- uses `LLMAgent`
- can also trigger short voice cues while building the result table

This is the best general-purpose UI for local demoing and clip review.

### B. `realtime_app.py`

Use this for:

- true live webcam or local video processing with OpenCV
- immediate on-screen warnings
- immediate offline spoken coaching cues

Key characteristics:

- local desktop OpenCV window
- webcam-friendly
- best fit when the camera and audio output are on the same local machine

This is the best choice for a **local laptop live coaching demo**.

### C. `web_realtime_app.py`

Use this for:

- browser webcam streaming
- live frame-by-frame analysis from a web page

Key characteristics:

- browser webcam input
- Gradio streaming UI
- useful when compute and camera are on different machines

This is the best choice for:

- browser-based live demo
- workstation backend plus laptop webcam

But note:

- voice output still happens on the backend machine because `pyttsx3` runs there

## 4. Core Modules and Responsibilities

### Inference stack

The inference package has moved from `src/inference` to top-level `inference/`.

Important files:

- `inference/sliding_window.py`
- `inference/activity_gatekeeper.py`
- `inference/target_tracker.py`
- `inference/video_annotator.py`
- `inference/heuristics_engine.py`

Current reality:

- this top-level `inference/` package is now the active one
- any older assumptions that the runtime uses `src/inference/*` are stale

### Voice coaching

- `realtime_voice_coach.py`
- `coach_playbook.json`

This is the new immediate-coaching path.

It does not depend on Gemini.
It maps machine-readable `error_key`s to:

- `error_name`
- `diagnosis`
- `short_cue`

and uses `pyttsx3` for offline text-to-speech.

### LLM summary

- `llm_agent.py`

This is still optional.

It uses:

- `google-genai`
- `GEMINI_API_KEY`
- model string currently set to `gemini-3.1-flash-lite-preview`

If the key is missing, `llm_agent.py` still writes a deterministic
`coach_playbook.json` summary that lists each detected problem and frequency.
If the key is present, Gemini receives the same playbook names, diagnoses,
short cues, and frequencies as part of its prompt.
The clip Analysis UI exposes this as a `Use Gemini Summary` checkbox so a user
can choose Gemini or the playbook-only summary per run.

### Persistence

- `database.py`
- `fencing_coach.db`

Tables:

- `Users`
- `Sessions`
- `ActionLogs`

## 5. What We Were Just Doing

Before this handoff, we:

1. pulled the latest `main`
2. inspected the newly added real-time modules
3. confirmed that the docs were behind the code
4. confirmed that local laptop usage is the right choice for webcam + voice
5. started cleaning the docs to match the current codebase
6. verified the local webcam can open and read frames
7. downloaded `yolov8n-pose.pt` for local YOLO pose inference
8. installed/listed `lap` for Ultralytics ByteTrack
9. made live target locking wait for the first usable detection
10. made the live pose path fall back cleanly if ByteTrack dependencies are missing
11. verified the local vision-only FFD folder at `FFD/FFD`
12. updated FFD preparation to auto-detect per-class video folders

The immediate handoff goal is not a new feature. It is to make local usage and next-step debugging much easier.

## 6. Known Mismatches and Risks

### A. Old docs used to describe removed or outdated flows

Examples of stale assumptions that were present before cleanup:

- `app.py` as a CLI
- `web_app.py` as the main browser demo
- `src/inference/*` as the active runtime inference package

The next agent should trust:

- current code
- this handoff
- updated quickstart

more than older historical prose.

### B. Left-handedness is still risky

The runtime still appears to assume a right-handed mapping in some pose handling paths.

Risk:

- `front_wrist`
- `front_ankle`

may still be biased toward the right side in some contexts.

This matters if the next agent wants robust local demos with left-handed fencers.

### C. Target lock is still simple, but no longer frame-0-only

`TargetTracker` now locks onto the first usable detection and follows the
ByteTrack ID when one exists. If persistent IDs are unavailable, it falls back
to a source-rank lock so fresh installs and mock smoke tests still run.

Risk:

- wrong target at first valid detection
- difficult recovery after tracking drift or ID changes

### D. Old tests are likely stale

Earlier inspection showed many tests failing after the architecture shift because the tests still expected:

- old CLI behavior
- old `src.app_interface` modules
- older tracking schema assumptions

This means:

- failing old tests do not automatically mean the new runtime is broken
- but the test suite needs to be realigned

### E. Gemini model string may need stabilization

The code currently uses a preview model name:

- `gemini-3.1-flash-lite-preview`

If local usage shows model-name or availability errors, the next agent should consider switching to a current stable Gemini model listed in official docs.

### F. Local FFD is vision-only

The local `FFD/FFD` folder contains per-class video clips, not Kinect
`*_Body.mat` files.

Current checked counts:

- `SF`: 24
- `SB`: 23
- `R`: 20
- `IS`: 19, from folder `3_IR`
- `WW`: 23
- `JS`: 21

The raw `.mov` recordings and `6_Idle` clips are ignored by the current
training prep because the model head is six-class.

## 7. Local Usage Recommendation

For a local laptop setup:

### Recommended first choice

Use:

- `python realtime_app.py --source 0 --mode "Free Bouting" --target-side left`

if the goal is:

- direct webcam
- direct local spoken feedback
- low-friction live demo

### Recommended second choice

Use:

- `python app.py`

if the goal is:

- recorded clip analysis
- session history
- LLM summary
- Gradio UI

### Browser streaming choice

Use:

- `python web_realtime_app.py`

if the goal is:

- browser webcam streaming
- visual-only or browser-oriented demo

## 8. Environment Notes for Local Setup

Likely requirements:

- Python venv
- `pip install -r requirements.txt`
- local YOLO pose weights such as `yolov8n-pose.pt`
- `lap` for Ultralytics ByteTrack persistent IDs
- optional `.env` file with `GEMINI_API_KEY`

For local voice coaching:

- `pyttsx3` must install and work on the local OS audio stack

If `lap` is missing, `realtime_app.py` falls back to plain pose detections
without persistent track IDs instead of crashing.

## 9. Suggested Next Tasks for the New Agent

1. verify local setup and entrypoints
2. test `realtime_app.py` on the local laptop webcam
3. test `app.py` on a short local video clip
4. verify that `pyttsx3` voice cues are audible locally
5. verify Gemini summary after `.env` is configured
6. inspect handedness handling for left-handed fencers
7. decide whether to stabilize or update stale tests

## 10. Practical Priorities

If time is limited, the next agent should prioritize:

1. **local runnability**
2. **correct docs**
3. **left-handed and target-tracking reliability**
4. **test-suite cleanup**

Not first priority:

- training pipeline changes
- broad research doc editing
- refactoring unrelated historical modules

## 11. Files to Read First

Best entry files for the next agent:

- `README.md`
- `docs/dev/QUICKSTART.md`
- `docs/dev/CURRENT_STATUS.md`
- `docs/dev/NEXT_AGENT_PROMPT.md`
- `app.py`
- `realtime_app.py`
- `web_realtime_app.py`
- `realtime_voice_coach.py`
- `coach_playbook.json`
- `inference/heuristics_engine.py`

## 12. Bottom Line

The project has recently shifted into a more usable local-demo shape.

The biggest need right now is not a brand-new model feature. It is:

- accurate local setup
- accurate runtime docs
- stable live demo behavior

That is the context the next agent should inherit.
