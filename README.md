# 🤺 AI Fencing Coach MVP

A real-time AI fencing coach that uses **YOLOv8 pose estimation** to track two fencers via webcam, evaluate distance heuristics, and deliver **spoken audio warnings** when fencers get too close — all running natively on your laptop with zero cloud dependencies.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green)
![YOLOv8](https://img.shields.io/badge/YOLOv8n--pose-Ultralytics-purple)

---

## ✨ Features

| Feature | Description |
|---|---|
| **Real-time Pose Tracking** | YOLOv8n-pose detects and draws full COCO skeletons on both fencers |
| **Fencer L/R Identification** | Automatically assigns Left/Right IDs based on X-axis position |
| **Dynamic Distance Threshold** | Scale-invariant — threshold adapts to camera distance using fencer bounding box height |
| **Audio Coaching** | Asynchronous text-to-speech warns when fencers are too close (non-blocking) |
| **Visual HUD** | On-screen distance, threshold, FPS counter, and red "TOO CLOSE!" flash warning |

---

## 📋 Prerequisites

- **Python 3.10+**
- A **webcam** (built-in laptop cam or USB)
- **Windows** (pyttsx3 uses SAPI5; macOS/Linux may need additional TTS backends)

---

## 🚀 Quick Start

### 1. Clone or download this project

```bash
cd path/to/hci
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

This installs:
- `opencv-python` — webcam capture and frame rendering
- `ultralytics` — YOLOv8n-pose model for multi-person pose estimation
- `pyttsx3` — offline text-to-speech engine

> **Note:** The YOLOv8n-pose model weights (`yolov8n-pose.pt`, ~6 MB) are downloaded automatically on first run.

### 3. Run the application

```bash
python app.py
```

### 4. Usage

1. A window titled **"AI Fencing Coach"** will open showing your webcam feed.
2. Stand two people in front of the camera (side view works best, like a fencing strip).
3. The system will:
   - Draw **skeletons** and **bounding boxes** on both detected fencers
   - Label them **Fencer L** (green, left side) and **Fencer R** (orange, right side)
   - Display **Dist / Thresh** values at the top center of the screen
   - Flash **"TOO CLOSE!"** in red and speak an audio warning when the distance rule triggers
4. Press **`q`** to quit and release the camera.

---

## ⚙️ Configuration

All tunable parameters are at the top of `app.py`:

| Variable | Default | Description |
|---|---|---|
| `YOLO_CONF` | `0.5` | Minimum detection confidence for YOLO |
| `KP_CONF` | `0.4` | Minimum keypoint confidence to draw/use a joint |
| `DISTANCE_MULTIPLIER` | `1.0` | "Too close" triggers when ankle distance < `multiplier × avg fencer height`. Lower = stricter, higher = more lenient |
| `AUDIO_COOLDOWN` | `4.0` | Seconds between consecutive audio warnings |

### Tuning the Distance Rule

The threshold is **proportional to the fencers' bounding box height**, making it camera-distance agnostic:

```
dynamic_threshold = avg_fencer_height × DISTANCE_MULTIPLIER
```

- `DISTANCE_MULTIPLIER = 0.8` → triggers when fencers are very close
- `DISTANCE_MULTIPLIER = 1.0` → triggers at roughly one body-height apart (default)
- `DISTANCE_MULTIPLIER = 1.5` → triggers earlier, more conservative

---

## 📁 Project Structure

```
hci/
├── app.py              # Main application (video loop, YOLO, rules engine, TTS)
├── requirements.txt    # Python dependencies
├── mvpspec.md          # Full project specification
└── README.md           # This file
```

---

## 🔧 How It Works

```
Webcam Frame
    │
    ▼
YOLOv8n-pose inference (conf > 0.5)
    │
    ▼
Filter to top-2 largest bounding boxes
    │
    ▼
Sort by center-X → assign Fencer L / Fencer R
    │
    ▼
Extract ankle keypoints (indices 15, 16)
    │
    ├─► engagement_dist = |front_ankle_L_x − front_ankle_R_x|
    ├─► dynamic_threshold = avg_bbox_height × DISTANCE_MULTIPLIER
    │
    ▼
Rules Engine: if dist < threshold → speak_async("Warning...")
    │
    ▼
Draw skeletons, boxes, labels, HUD → cv2.imshow
```

---

## 🐛 Troubleshooting

| Issue | Solution |
|---|---|
| `ModuleNotFoundError: No module named 'cv2'` | Run `pip uninstall opencv-python-headless -y && pip install opencv-python` |
| `numpy.dtype size changed` | Run `pip install --upgrade numpy pandas` |
| No audio on macOS/Linux | `pyttsx3` defaults to SAPI5 (Windows). On Linux install `espeak`: `sudo apt install espeak` |
| Low FPS | Try reducing camera resolution in `app.py` (change 1280×720 to 640×480) |
| Only 1 fencer detected | Ensure both fencers are fully visible; adjust `YOLO_CONF` lower if needed |

---

## 📝 License

This project is for educational / HCI research purposes.
