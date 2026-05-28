# Android On-Device App

Native Android v1 for the AI Fencing Coach. The app runs live coaching on the
phone:

```text
CameraX frame
-> selected pose backend (MediaPipe or YOLO)
-> skeleton mapping
-> target lock + short-gap interpolation
-> SpatialNormalizer + FenceNet ONNX
-> Kotlin heuristics + feedback scheduler
-> overlay + Android TextToSpeech cue
-> post-practice review
```

## Setup

1. Install Python 3.11 or 3.12. Do not use Python 3.14 for the export
   environment; NumPy/PyTorch/ONNX wheels may not exist for it yet on Windows.
2. Create and activate a venv from the repo root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

3. Open this `android/` folder in Android Studio.
4. Let Android Studio install the Android SDK/Gradle files it recommends.
5. Generate the ONNX assets from the repo root:

```powershell
python scripts/export_fencenet_onnx.py
python scripts/export_yolo_pose_onnx.py
```

6. Confirm these files exist:

```text
android/app/src/main/assets/fencenet_v2.onnx
android/app/src/main/assets/pose_landmarker_lite.task
android/app/src/main/assets/yolo_pose.onnx
android/app/src/main/assets/coach_playbook.json
```

The MediaPipe lite pose model is downloaded from:

```text
https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task
```

## Run

Use a physical Android phone. The emulator is not a meaningful test target for
camera latency.

From Android Studio:

1. Select the `app` run configuration.
2. Connect a phone with USB debugging enabled.
3. Run the app.
4. Point the back camera at a side-view fencing stance.

## Current V1 Scope

- Realtime live coach is the only fully connected Android runtime path.
- No runtime backend.
- App opens to a home screen with Realtime, Postgame, and User Settings.
- Realtime has a compact setup screen that reads training mode, pose model, target side, voice, and feedback focus from User Settings before opening the camera.
- The live coaching screen is camera-first: preview and skeleton overlay on top, current error/status and controls below.
- Postgame reads clip-analysis and summary defaults from User Settings, runs selected videos through the on-device pose/FenceNet/heuristics pipeline, shows a processing progress bar, and displays the generated report.
- User Settings is scrollable and includes user information, app defaults, Gemini/playbook summary preference, and per-error emphasize/mute checkboxes.
- Pose backend selector in the HUD.
- MediaPipe backend uses the MediaPipe Tasks pose landmarker.
- YOLO backend runs `yolo_pose.onnx` through ONNX Runtime with local decoding/NMS.
- Target tracker keeps the selected fencer locked through short pose dropouts.
- FenceNet only receives active fencing frames; idle frames do not fill the model window.
- HUD shows target lock, warmup progress, cue stack/history, FPS, latency, dropped-frame estimate, and session counts.
- Controls include training mode, pose backend, target side, pause/resume, voice, finish, reset, and menu.
- Post-practice review summarizes time, active time, model checks, top action, repeated cues, and recent cue timeline.
- Back camera by default.
- Landscape orientation.
- One selected target fencer plus optional opponent context.

Clip review, summaries, cloud sync, and iOS are intentionally later phases.
