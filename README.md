# 🤺 AI Fencing Coach & Referee System

A comprehensive AI-powered fencing coaching and referee assistance system built on **FenceNet** (Fine-grained 2D skeleton-based footwork recognition) and open-source multimodal LLMs. This system provides real-time analysis, tactical coaching, and long-term athlete progression tracking.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green)

---

## ✨ Core Features

| Feature | Description |
|---------|-------------|
| **Real-time Skeleton Detection** | Extract 2D poses from video using state-of-the-art pose estimators |
| **Footwork Classification** | Recognize 6 fencing actions: R, IS, WW, JS, SF, SB |
| **Pattern Analysis** | Calculate offensive ratios, defensive metrics, JS/SF ratios |
| **Virtual Coach AI** | LLM-powered feedback (immediate, strategic, conclusive) |
| **Athlete Profiles** | Long-term performance tracking and progression analysis |
| **Interactive Dashboard** | Real-time UI showing video, classifications, and coaching insights |
| **Batch Processing** | Process videos frame-by-frame with sliding window inference |
| **GPU Acceleration** | Full CUDA support for fast inference |

---

## 🏗️ System Architecture

The system operates in 6 integrated stages:

```
┌─────────────────────────────────────────────────────────┐
│  Phase 1: Pose Estimation  → 2D Skeleton Keypoints    │
├─────────────────────────────────────────────────────────┤
│  Phase 2: Preprocessing    → Normalization + Sampling  │
├─────────────────────────────────────────────────────────┤
│  Phase 3: FenceNet         → Action Classification     │
├─────────────────────────────────────────────────────────┤
│  Phase 4: Pattern Tracking → Statistical Analysis      │
├─────────────────────────────────────────────────────────┤
│  Phase 5: LLM Coaching     → Feedback Generation       │
├─────────────────────────────────────────────────────────┤
│  Phase 6: Interactive UI   → Real-time Dashboard       │
└─────────────────────────────────────────────────────────┘
```

---

## 📊 Target Fencing Actions

The system classifies 6 fundamental fencing footwork patterns:

| Code | Action | Category |
|------|--------|----------|
| **R** | Rapid Lunge | Offensive |
| **IS** | Incremental Speed Lunge | Offensive |
| **WW** | With Waiting Lunge | Offensive |
| **JS** | Jumping Sliding Lunge | Offensive |
| **SF** | Step Forward | Neutral |
| **SB** | Step Backward | Defensive |

---

## 📋 Prerequisites

- **Python 3.10+**
- **PyTorch 2.0+** with CUDA support (optional but recommended)
- **CUDA 11.8+** for GPU acceleration (optional)
- 4GB RAM minimum (8GB+ recommended for better performance)
- **Video input**: webcam, video file, or recorded bout footage

---

## 🚀 Quick Start

### 1. Clone or download the project

```bash
cd ai-fencing-coach-mvp
```

### 2. Create a virtual environment (recommended)

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

This installs:
- `torch` & `torchvision` — Deep learning framework
- `opencv-python` — Video processing and UI
- `transformers` — LLM model loading
- `numpy`, `pandas` — Data handling

### 4. Run the application

**Interactive mode (recommended for first-time use):**
```bash
python app.py --interactive
```

**Process a single video:**
```bash
python app.py --video path/to/bout.mp4 --fencer-id athlete_001 --fencer-name "Smith"
```

**Use BiFenceNet (bidirectional analysis):**
```bash
python app.py --video bout.mp4 --use-bifencenet --fencer-id athlete_001
```

**Specify GPU acceleration:**
```bash
python app.py --video bout.mp4 --device cuda --fencer-id athlete_001
```

---

## 📁 Project Structure

```
ai-fencing-coach-mvp/
├── app.py                              # Main entry point
├── requirements.txt                    # Python dependencies
├── README.md                           # English documentation (this file)
├── README_zh.md                        # Chinese documentation
├── detailedstructure.md                # Full system specification
└── src/
    ├── pose_estimation/                # [Phase 1] Skeleton extraction
    │   └── pose_estimator.py
    ├── preprocessing/                  # [Phase 2] Normalization & sampling
    │   ├── spatial_normalizer.py
    │   └── temporal_sampler.py
    ├── models/                         # [Phase 3] FenceNet architecture
    │   ├── tcn_block.py
    │   ├── fencenet.py
    │   └── bifencenet.py
    ├── tracking/                       # [Phase 4] Pattern analysis
    │   ├── pattern_analyzer.py
    │   └── profile_manager.py
    ├── llm_agent/                      # [Phase 5] Virtual coach engine
    │   ├── prompt_templates.py
    │   ├── model_loader.py
    │   └── coach_engine.py
    └── app_interface/                  # [Phase 6] Orchestration & UI
        ├── system_pipeline.py
        └── main_ui.py
```

---

## 🔧 How It Works

```
Video Input
    │
    ▼
[Phase 1] Pose Estimation → 2D skeleton keypoints
    │
    ▼
[Phase 2] Preprocessing → Spatial normalization + temporal sampling to 28 frames
    │
    ▼
[Phase 3] FenceNet Inference → Action classification (R, IS, WW, JS, SF, SB)
    │
    ▼
[Phase 4] Pattern Analysis → Calculate metrics (offensive ratio, JS/SF ratio, etc.)
    │
    ▼
[Phase 5] LLM Coaching → Generate contextual feedback
    │
    ├─► Immediate Feedback (real-time)
    ├─► Break Strategy (tactical advice)
    └─► Conclusive Analysis (post-bout)
    │
    ▼
[Phase 6] Interactive UI → Display results + athlete profile update
    │
    ▼
    Athlete Profile → JSON storage
```

---

## 📊 Metrics & Statistics

The system calculates:

| Metric | Description |
|--------|-------------|
| **Offensive Ratio** | Percentage of time fencer uses offensive actions (R, IS, WW, JS) |
| **Defensive Ratio** | Percentage of time fencer uses defensive actions (SB) |
| **JS/SF Ratio** | Jumping Sliding Lunge vs Step Forward ratio (footwork efficiency) |
| **Action Frequency** | Percentage breakdown of each of 6 actions |
| **Patterns** | Repetitive action sequences and transitions |
| **Confidence** | Average model confidence in predictions |

These metrics are saved per bout and used for LLM feedback generation.

---

## 🎯 Use Cases

### 1. Real-time Coaching
- Coach watches athlete during practice
- Receives immediate tactical feedback every few seconds
- System identifies patterns and suggests adjustments

### 2. Post-Bout Analysis
- Upload bout footage to system
- Get comprehensive performance summary
- Compare with historical data
- Identify strengths and weaknesses

### 3. Long-term Progression Tracking
- System maintains athlete profile
- Tracks win rates, action preferences, tactical evolution
- Identifies improvement trends over multiple bouts

### 4. Opponent Analysis
- Upload opponent's previous bouts
- System learns their typical patterns
- Coach receives tactical recommendations before facing them

---

## 🔑 Command Reference

```bash
# Interactive mode
python app.py --interactive

# Process video with defaults
python app.py --video bout.mp4

# Full options
python app.py \
  --video bout.mp4 \
  --fencer-id fencer_001 \
  --fencer-name "John Smith" \
  --opponent-id fencer_002 \
  --opponent-name "Jane Doe" \
  --use-bifencenet \
  --device cuda \
  --model weights/fencenet/model_best.pth

# Help
python app.py --help
```

---

## ⚡ Performance Tips

1. **GPU Usage**: Use `--device cuda` for 10-50x speedup
2. **Model Size**: BiFenceNet is larger but more accurate
3. **Batch Processing**: The system processes frames in batches automatically
4. **Memory**: Reduce batch size if running out of RAM

---

## 📚 Documentation

- **[detailedstructure.md](detailedstructure.md)** - Full technical specification
- **[README_zh.md](README_zh.md)** - Chinese documentation
- Source code comments - Detailed explanations in each module

---

## 🔄 Development Workflow

```bash
# Clone and setup
git clone <repo>
cd ai-fencing-coach-mvp
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run tests
pytest

# Format code
black src/

# Check style
flake8 src/
mypy src/

# Run application
python app.py --interactive
```

---

## 🚀 Future Enhancements

- [ ] Real-time webcam streaming support
- [ ] Multi-athlete simultaneous tracking
- [ ] Video recording and playback features
- [ ] Web-based dashboard (Flask/Vue.js)
- [ ] Model fine-tuning tools
- [ ] Advanced statistical analysis
- [ ] Export reports (PDF/Excel)
- [ ] Mobile app (iOS/Android)

---

## 📝 License

MIT License - See LICENSE file for details

---

## 📧 Contact & Support

For questions, issues, or feature requests, please visit the project repository or contact the development team.

---

**Version**: 0.1.0  
**Last Updated**: April 7, 2026  
**Python**: 3.10+  
**PyTorch**: 2.0+

| `ModuleNotFoundError: No module named 'cv2'` | Run `pip uninstall opencv-python-headless -y && pip install opencv-python` |
| `numpy.dtype size changed` | Run `pip install --upgrade numpy pandas` |
| No audio on macOS/Linux | `pyttsx3` defaults to SAPI5 (Windows). On Linux install `espeak`: `sudo apt install espeak` |
| Low FPS | Try reducing camera resolution in `app.py` (change 1280×720 to 640×480) |
| Only 1 fencer detected | Ensure both fencers are fully visible; adjust `YOLO_CONF` lower if needed |

---

## 📝 License

This project is for educational / HCI research purposes.
