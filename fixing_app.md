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
The current `FenceNetV2` model accepts a fixed input shape of `(28, 18)` after pose preprocessing.

* **Input**: Normalized skeleton matrix of shape `(T, 9, 2)` where `T` is the total number of frames in the video.
* **Parameters**:
    * `WINDOW_SIZE` = 28 frames
    * `STRIDE` = 10 frames (~0.33 seconds, the step size for the sliding window)
* **Smoothing Logic (Non-Maximum Suppression / NMS)**:
    1.  At each step, the model outputs a predicted label and a Confidence Score (via Softmax).
    2.  If the Confidence Score is `< 0.6`, label the window as `Idle` (ignore).
    3.  For `SF` / `SB`, apply a motion sanity check from the target fencer's horizontal displacement. If the predicted step direction contradicts the observed motion, flip the label before timeline generation.
    4.  Resolve all overlapping windows into a single non-overlapping timeline by keeping the highest-confidence label at each frame, then collapse contiguous runs into final `action_segments`.
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
    - Action Stats: ["SF", "SF", "SB", "R", "IS", "WW", "SF", "R" , "SB", "JS"]
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
  * The system may also accept an optional `weapon_hand` override (`"auto"`, `"left"`, `"right"`) for the selected target fencer.
  * On the first valid tracked frame, evaluate the bounding box center X-coordinates of all detected persons.
  * If `target_side == "left"`, lock onto the `track_id` with the minimum X-coordinate.
  * If `target_side == "right"`, lock onto the `track_id` with the maximum X-coordinate (excluding obvious background persons/referees by checking bounding box size/aspect ratio if necessary).
* **Data Filtering**:
  For the rest of the video, ONLY append the skeletal data of the locked `track_id` to the time-series array. If the `track_id` is temporarily lost due to occlusion, pad with the last known coordinates for small gaps (<= 5 frames). If YOLO/ByteTrack reassigns the ID, the tracker should relock to the nearest plausible candidate instead of freezing permanently on the stale ID.
* **Posture Engine Adaptation (Rule update)**:
  Pass the `target_side` to the pose / tracking path so it can canonicalize the target skeleton into `front_shoulder`, `front_elbow`, `front_wrist`, and `front_ankle`.
  * `target_side` means which athlete is on the left or right side of the screen.
  * If `weapon_hand == "auto"`, the system should use that screen-side cue to decide which anatomical arm / leg is actually leading in the current pose, rather than hard-coding the right arm as the weapon arm.
  * If `weapon_hand == "left"` or `"right"`, force the target fencer's front arm / leg to that anatomical side.
  * Downstream geometry checks should trust the canonicalized `front_*` joints when determining the front leg and weapon hand.

### Module 5: Activity Gatekeeper & State Machine (Idle vs. Active)
To prevent computational waste and hallucination during non-fencing (resting/walking) periods in a continuous camera feed, implement a lightweight geometric gatekeeper *before* the Sliding Window.

* **State Machine Definition**:
  * System starts in `State: IDLE`.
  * If `IDLE`: Only run YOLO extraction at 5 FPS (skip frames) to save compute. Do NOT feed data to FenceNet.
  * If `ACTIVE`: Run YOLO at full source-video FPS and feed data to the Sliding Window (Module 1).
* **Transition Logic (Heuristic-based)**:
  * **To ACTIVE**: Calculate the Knee Angle of the target fencer. If `Knee_Angle < 155°` (indicating the fencer has dropped into the "En Garde" stance) and the bounding box is moving, switch to `ACTIVE`. Require this condition to hold for at least 15 consecutive frames (0.5s) to prevent false triggers.
  * **To IDLE**: If the fencer stands up (`Knee_Angle > 165°`) or turns their back to the camera (shoulder width approaches 0 due to 2D projection), OR if the distance between the two fencers exceeds 60% of the frame width, switch to `IDLE` after a 2-second cooldown.
* **Logging**: 
  The final JSON report should only include timestamps where the system was in the `ACTIVE` state.

### Module 6: Prediction Validation UI (Model Debugging Interface)

**Objective**: Build a rapid prototyping Web UI to visually validate the Sliding Window predictions and Target Tracking accuracy. The UI must ensure 100% perfect synchronization between the video frames and the model's output.

* **Recommended Framework**: `Gradio` or `Streamlit` (for rapid Python UI development).
* **Core Workflow**:
  1. **Upload**: User uploads a raw fencing video (`.mp4`).
  2. **Parameter Selection**: User selects `target_side` (Left / Right Fencer).
  3. **Processing**: System runs Module 4 (Tracking), Module 5 (Gatekeeper), and Module 3 (FenceNet).
  4. **Burn-in Rendering**: System uses OpenCV to generate an annotated output video.
  5. **Playback**: The UI displays the annotated video alongside a downloadable JSON log.

* **Video Annotation Requirements (OpenCV `cv2.putText` & `cv2.rectangle`)**:
  To guarantee exact temporal synchronization, the predictions must be rendered directly onto the output video frames.
  The annotation layer must index tracking metadata by the original `frame_index`, because Gatekeeper skipping makes the tracking timeline sparse during `IDLE`.
  The saved JSON report must preserve `target_side` and `weapon_hand`, otherwise the oldstyle single-target debug overlay cannot reliably re-select the same athlete offline.
  * **Bounding Box**: Draw a bounding box around the selected `track_id` (the target fencer).
  * **Skeleton Overlay**: Draw the YOLO pose connections for the target fencer to verify tracking quality.
  * **Status HUD (Heads-Up Display)**: In the top-left corner of the video, display:
    * `State`: `ACTIVE` (Green) or `IDLE` (Red) - from the Gatekeeper.
    * `Knee Angle`: Display current knee angle to verify the En Garde check.
    * `Current Action`: The best action covering the current frame. If multiple segments overlap, prefer the strongest-confidence segment instead of the earliest one.
    * `Confidence`: `0.92` (Only display if `State` is `ACTIVE`).

* **UI Layout Components**:
  * **Input Section**: `File Uploader` and `Radio Buttons` (Target Fencer: Left/Right).
  * **Action Button**: `Run Analysis`.
  * **Output Section 1 (Visual)**: A `Video Player` component playing the annotated `.mp4`.
* **Output Section 2 (Data)**: A `Dataframe/Table` showing the parsed action timeline (e.g., Start Time, End Time, Action Label) for easy cross-referencing.

---

## 2.1 Current Implementation Notes

- The local `FFD.zip` archive is the authoritative source for the training clips used by `convert_to_json.py`.
- The converter must handle both `3_IS/` and `3_IR/` as the `IS` class.
- The converter should canonicalize `front_*` joints from the clip's screen side before saving JSON; otherwise the model will learn the wrong weapon-hand mapping.
- The Gradio validation UI should use the sparse-frame-safe annotation path; otherwise later video frames can appear unannotated even when the backend is still processing them.
- The oldstyle single-target debug video should read the final non-overlapping `action_segments`, not the first overlapping window that happens to cover the current frame.
- Final `SF` / `SB` segments should be rechecked against tracked target motion in pixel space before writing the report, so the offline debug video matches what the user visually sees.
- `mock` pose mode is for smoke testing the pipeline structure and should not emulate Gatekeeper frame skipping.
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
