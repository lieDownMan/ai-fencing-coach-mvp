# Android Model Assets

Required runtime assets:

- `pose_landmarker_lite.task`
- `fencenet_v2.onnx`
- `coach_playbook.json`

Generate `fencenet_v2.onnx` from the repo root:

```powershell
python scripts/export_fencenet_onnx.py
```

Download the MediaPipe lite pose model into this folder:

```powershell
Invoke-WebRequest `
  -Uri "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task" `
  -OutFile "android/app/src/main/assets/pose_landmarker_lite.task"
```
