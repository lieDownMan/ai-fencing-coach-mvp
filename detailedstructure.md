# Project Specification: AI Fencing Coach & Referee System

## 1. Project Overview
This project implements a comprehensive remote fencing coaching and referee assistance system. It utilizes `FenceNet` for fine-grained 2D skeleton-based footwork recognition and integrates an Open-Source Multimodal Large Language Model (e.g., LLaVA-NeXT, Qwen2-VL) to act as a virtual coach. The system monitors bouts, analyzes fencing patterns, provides tiered feedback (immediate, inter-bout breaks, and post-bout conclusions), and tracks long-term athlete progression.

## 2. Directory Structure
```text
fencing_coach_system/
├── data/
│   ├── videos/                # Raw bout videos
│   └── fencer_profiles/       # JSON/DB files storing historical patterns and progression
├── weights/
│   ├── pose_estimator/        # 2D Pose model weights (e.g., YOLO-Pose)
│   ├── fencenet/              # Trained FenceNet/BiFenceNet weights
│   └── llm_models/            # Local Open-Source LLM weights (e.g., Qwen2-VL, LLaVA)
├── src/
│   ├── pose_estimation/       # Phase 1: Video to 2D Skeleton
│   ├── preprocessing/         # Phase 2: Skeleton Normalization & Sampling
│   ├── models/                # Phase 3: FenceNet Footwork Classification
│   ├── tracking/              # Phase 4: Pattern Analysis & Progression Tracking
│   │   ├── __init__.py
│   │   ├── pattern_analyzer.py# Aggregates FenceNet outputs (e.g., calculates JS vs SF ratio)
│   │   └── profile_manager.py # Reads/Writes athlete stats to 'data/fencer_profiles/'
│   │
│   ├── llm_agent/             # Phase 5: The Virtual Coach Engine
│   │   ├── __init__.py
│   │   ├── prompt_templates.py# System prompts for Immediate, Break, and Conclusive feedback
│   │   ├── model_loader.py    # Inference pipeline for the local open-source LLM
│   │   └── coach_engine.py    # Injects tracking stats and context into prompts to generate advice
│   │
│   └── app_interface/         # Phase 6: System Orchestration & HCI
│       ├── __init__.py
│       ├── system_pipeline.py # Orchestrates Pose -> FenceNet -> Tracking -> LLM
│       └── main_ui.py         # Real-time dashboard for fencers/coaches
├── requirements.txt           
└── README.md
```

## 3. Target Classes
The FenceNet classification layer must output logits for the following 6 fencing footwork actions:
1. `R`: Rapid lunge
2. `IS`: Incremental speed lunge
3. `WW`: With waiting lunge
4. `JS`: Jumping sliding lunge
5. `SF`: Step forward
6. `SB`: Step backward

## 4. Module Specifications

### 4.1. Vision Pipeline (`src/pose_estimation/`)
* **Responsibility**: Run an off-the-shelf 2D pose estimator on input videos.
* **Constraint**: Must extract the x, y coordinates for the following joints: front wrist, front elbow, front shoulder, both hips, both knees, and both ankles. The nose and front ankle are also strictly required for spatial normalization.

### 4.2. Preprocessing Pipeline (`src/preprocessing/`)
* **Spatial Normalizer**: 
  * Subtract the fencer's nose position of the *first frame* from every joint coordinate in each frame.
  * Divide each coordinate by the vertical distance between the head position and front ankle in the *first frame*.
* **Temporal Sampler**: Extract or sample sequences into exactly 28 consecutive frames to ensure uniform input length for the model. 

### 4.3. FenceNet Architecture (`src/models/`)
* **TCN Blocks**: Implement Temporal Convolutional Network (TCN) blocks. Must include causal convolutions to prevent future data leakage and dilated convolutions to increase the receptive field. Include a residual connection `output = Activation(p + f(p))`.
* **FenceNet**: Stack 6 TCN blocks. Extract the last time-step from the output sequence, feeding it into dense layers for prediction.
* **BiFenceNet**: Implement a bidirectional module containing a causal network (forward motion) and an anti-causal network (reversed motion). Concatenate their last time steps before the dense layers.

### 4.4. Data Tracking & Pattern Analysis (`src/tracking/`)
* **`pattern_analyzer.py`**: Ingests real-time classifications from FenceNet. Calculates statistical metrics (e.g., action frequencies, defensive vs. offensive ratios, reaction delays). Identifies repetitive patterns.
* **`profile_manager.py`**: Saves match statistics and bout summaries to individual fencer profiles to trace long-term technical progression.

### 4.5. LLM Coaching Engine (`src/llm_agent/`)
* **Responsibility**: Acts as the remote coaching system, generating actionable intelligence using a local open-source MLLM.
* **`prompt_templates.py`**: Implement three specific feedback loops:
  1. **Immediate Feedback**: Extremely concise feedback delivered during the bout (e.g., "Shorten your recovery step").
  2. **Break Strategy**: Delivered during the 1-minute break. Summarizes opponent patterns and suggests tactical adjustments against the opponent.
  3. **Conclusive Advice**: Delivered after the bout. Compares current performance with historical `profile_manager` data to provide progression insights.
* **`coach_engine.py`**: Orchestrates the formatting of tracking data and MLLM inference calls.

### 4.6. Application Interface (`src/app_interface/`)
* **Responsibility**: Provides the Human-Computer Interaction dashboard.
* **UI Components**: Must display the live video feed with bounding boxes, a real-time footwork classification log, and a dedicated "Coach's Corner" panel displaying the MLLM-generated feedback.