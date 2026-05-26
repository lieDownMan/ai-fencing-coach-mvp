# Android On-Device App

Native Android v1 for the AI Fencing Coach. The app runs live coaching on the
phone:

```text
CameraX frame
-> MediaPipe Pose Landmarker
-> skeleton mapping
-> SpatialNormalizer + FenceNet ONNX
-> Kotlin heuristics + feedback scheduler
-> overlay + Android TextToSpeech cue
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
5. Generate the FenceNet ONNX asset from the repo root:

```powershell
python scripts/export_fencenet_onnx.py
```

6. Confirm these files exist:

```text
android/app/src/main/assets/fencenet_v2.onnx
android/app/src/main/assets/pose_landmarker_lite.task
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

- Live coach only.
- No runtime backend.
- Back camera by default.
- Landscape orientation.
- One selected target fencer plus optional opponent context.

Clip review, summaries, cloud sync, and iOS are intentionally later phases.
