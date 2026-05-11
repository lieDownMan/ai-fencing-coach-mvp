# Prompt for the Next Codex Agent

Use this prompt when starting a new local Codex session for this project.

```text
You are continuing work on the AI Fencing Coach project from a local laptop setup.

Start by reading these files first:
- docs/dev/CURRENT_STATUS.md
- docs/dev/QUICKSTART.md
- README.md
- app.py
- realtime_app.py
- web_realtime_app.py
- realtime_voice_coach.py
- coach_playbook.json
- inference/heuristics_engine.py

Current situation:
- The repo recently pulled new code from main with commit a755b8a ("add realtime module").
- The project now has three main runtime entrypoints:
  1. app.py for clip analysis in Gradio
  2. realtime_app.py for local OpenCV webcam + voice coaching
  3. web_realtime_app.py for browser webcam streaming
- The active inference package is now top-level inference/, not src/inference/.
- Voice coaching uses pyttsx3 and coach_playbook.json.
- Gemini summary is optional and depends on GEMINI_API_KEY in a local .env.

Primary goal:
- make the project work smoothly on the local laptop, especially webcam-based usage

Please do this in order:
1. verify which entrypoint is best for local webcam + local audio
2. run the most relevant smoke test or manual run for local usage
3. identify runtime blockers for local demo
4. fix the most important blockers
5. update docs if the actual runtime behavior differs from the docs

Things to pay attention to:
- handedness assumptions may still be right-hand biased
- target tracking still relies on a simple frame-0 lock
- old tests may be stale and still expect old app_interface or CLI structure
- docs were historically behind the code, so trust current code over old prose

If you need a first practical runtime target, prioritize:
- realtime_app.py on a local webcam

If that is unstable or inconvenient, fall back to:
- app.py with a short local clip

Do not spend the first round on research docs or training unless they directly block the local demo path.
```
