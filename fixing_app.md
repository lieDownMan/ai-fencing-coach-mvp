# AI Fencing Coach System - MVP Implementation Specification (v1.0)

**Date**: April 26, 2026  
**Objective**: Upgrade the single-action clip classification model (`best_model.pth`) into a fully automated AI coaching pipeline capable of processing continuous, long-form videos.

---

## 1. Architecture Overview

The system receives a raw, continuous video and outputs a JSON report containing a timeline of actions, action labels, and posture correction feedback. The pipeline is divided into four main modules:

1.  **Pose Extractor (YOLOv8)**: Converts the continuous video into a frame-by-frame 2D skeleton coordinate matrix.
2.  **Action Spotter (Sliding Window + FenceNet)**: Identifies *when* specific actions occur and *what* they are.
3.  **Posture Evaluator (Geometric Heuristics)**: Applies specific geometric math rules based on the identified action labels to evaluate posture correctness.
4.  **Feedback Generator (LLM)**: Consolidates the data to generate personalized, human-like coaching feedback.

---

## 2. Module Specifications

### Module 1: Sliding Window Inference
The current `FenceNetV2` model only accepts a fixed input shape of `(30, 24)`. This module must slide across the timeline to crop segments and filter out noise.

* **Input**: Normalized skeleton matrix of shape `(T, 24)` where `T` is the total number of frames in the video.
* **Parameters**:
    * `WINDOW_SIZE` = 30 frames (~1 second)
    * `STRIDE` = 10 frames (~0.33 seconds, the step size for the sliding window)
* **Smoothing Logic (Non-Maximum Suppression / NMS)**:
    1.  At each step, the model outputs a predicted label and a Confidence Score (via Softmax).
    2.  If multiple consecutive overlapping windows predict the same attacking action (e.g., `WW`, `JS`, `R`), retain only the window with the highest Confidence Score to represent the actual occurrence of that action.
    3.  If the Confidence Score is `< 0.6`, label the segment as `Idle` (ignore).
* **Output Format**: `List[Dict]`, for example:
    ```json
    [
      {"start_frame": 120, "end_frame": 150, "action": "SF", "confidence": 0.88},
      {"start_frame": 180, "end_frame": 210, "action": "R", "confidence": 0.92}
    ]
    ```

### Module 2: Geometric Evaluator (Posture Engine)
**This is the core logic for corrections.** This module executes *only* within the timeframes identified by Module 1, directly parsing the raw YOLO coordinates to perform geometric calculations.

**2.1 Math Utils**
Implement a function to calculate the angle between vectors. Given three joint coordinates $A, B, C$ (where $B$ is the vertex), calculate the angle $\theta$:

$$
\theta = \arccos\left(\frac{\vec{BA} \cdot \vec{BC}}{|\vec{BA}| |\vec{BC}|}\right) \times \frac{180}{\pi}
$$

**2.2 Rule-based Checks**
Apply different heuristic functions based on the `action` label:

* **Rule A: Lunge Over-extension Check (Knee over toes)**
    * **Trigger**: When `action` is `R`, `IS`, `WW`, or `JS`.
    * **Logic**: Identify the frame with the maximum displacement within the window (the point of maximum extension). Calculate the angle of the front leg using the `[Hip, Knee, Ankle]` joints.
    * **Threshold**: If $\theta < 90^\circ$, log the error: `{"error": "Knee over toes", "severity": "high"}`.
* **Rule B: Weapon Hand Height Check (Guard dropped)**
    * **Trigger**: Any action (`SF`, `SB`, or `Lunge` variations).
    * **Logic**: Calculate the relative Y-axis relationship between the "Front Wrist" and the "Nose" or "Pelvis Center" throughout the window.
    * **Threshold**: If the Wrist's Y-coordinate drops below the Pelvis (assuming Y-axis increases downwards), log the error: `{"error": "Guard dropped", "severity": "medium"}`.
* **Rule C: Center of Mass Bouncing Check**
    * **Trigger**: Step Forward (`SF`) or Step Backward (`SB`).
    * **Logic**: Calculate the standard deviation or max/min variance of the Pelvis Center's Y-coordinate over the 30 frames.
    * **Threshold**: If the vertical variance exceeds 10% of the fencer's bounding box height, log the error: `{"error": "Unstable center of mass", "severity": "medium"}`.

### Module 3: Feedback Generation
Aggregate the results from Modules 1 and 2, and call an LLM API (e.g., OpenAI, Gemini) to generate the final report.

* **Prompt Template**:
    ```text
    You are a professional fencing coach. Here is the objective data analysis of a student's 1-minute drill:
    - Action Stats: 5 Step Forwards, 8 Step Backwards, 2 Rapid Lunges.
    - Detected Posture Errors:
      1. During the 2nd Step Forward, the weapon hand dropped below the waist line.
      2. During the 1st Lunge, the front knee angle was less than 90 degrees (over-lunging).
    Provide a concise, encouraging, but direct technical correction and tactical advice in under 100 words.
    ```
### Module 4: Target Isolation & Tracking (CRITICAL FOR MATCH VIDEOS)
Since real fencing videos contain two fencers (and often a referee), the system must isolate the target fencer's skeleton stream before feeding it to the sliding window.

* **Tracking Implementation**: 
  Use YOLOv8's built-in tracker (`model.track(source, persist=True, tracker="bytetrack.yaml")`) to assign consistent `track_id`s to all detected persons across frames.
* **Target Selection Logic**:
  * The system must accept a parameter: `target_side` (either `"left"` or `"right"`).
  * At `Frame 0`, evaluate the bounding box center X-coordinates of all detected persons.
  * If `target_side == "left"`, lock onto the `track_id` with the minimum X-coordinate.
  * If `target_side == "right"`, lock onto the `track_id` with the maximum X-coordinate (excluding obvious background persons/referees by checking bounding box size/aspect ratio if necessary).
* **Data Filtering**:
  For the rest of the video, ONLY append the skeletal data of the locked `track_id` to the time-series array. If the `track_id` is temporarily lost due to occlusion, pad with the last known coordinates or use linear interpolation for small gaps (<= 5 frames).
* **Posture Engine Adaptation (Rule update)**:
  Pass the `target_side` to Module 2 (Geometric Evaluator) so it knows which side is the "front" limb. 
  * If `"left"`, the fencer faces right -> Front leg is Right Leg (Index 14, 16), Front arm is Right Arm (Index 8, 10).
  * If `"right"`, the fencer faces left -> Front leg is Left Leg (Index 13, 15), Front arm is Left Arm (Index 7, 9).
---

## 3. Agent Action Items

Please execute the development in the following phases and submit the corresponding PRs or scripts:

* **Phase 1: `sliding_window.py`**
    * Implement the window slicing logic.
    * Load `best_model.pth` for batch inference over the slices.
    * Implement Non-Maximum Suppression (NMS) to merge overlapping time segments and filter low-confidence noise.
* **Phase 2: `heuristics_engine.py`**
    * Implement the vector angle calculation utility.
    * Implement `Rule A`, `Rule B`, and `Rule C` checkers.
    * Build a Dispatcher that routes the data to the appropriate Rule Checker based on the input `action` string.
* **Phase 3: `coach_pipeline.py`**
    * Wrap the end-to-end flow: `Video -> YOLO -> FenceNet (Sliding) -> Heuristics -> JSON Report`.

