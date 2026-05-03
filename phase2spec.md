# AI Fencing Coach MVP - Phase 2 Implementation Specification

**Project Context**: 
We are building a computer vision pipeline for fencing analysis. We already have a trained classification model (`FenceNet`, expecting shape `(30, 24)`) and a YOLOv8-pose extractor. 
This phase focuses on implementing intent-based heuristic evaluation and a validation UI for human coaches to review the results.

---

## 1. User Intent Modes (動態練習模式)

The system must accept a `training_mode` parameter from the user before processing the video. This mode determines which heuristic rules are strictly enforced and which are relaxed.

*   **Mode A: Footwork (腳步練習)**
    *   **Target Actions**: Step Forward (`SF`), Step Backward (`SB`), Rapid Lunge (`R`), Jump sliding Lunge (`JS`), With Waiting lunge  (`WW`), incremental speed lunge(`IS`).
    *   **Active Rules**: Strict Center of Mass (CoM) stability, strict stance width and hand angle.
*   **Mode B: Target Practice (刺靶練習)**
    *   **Target Actions**: Rapid Lunge (`R`), Jump sliding Lunge (`JS`), With Waiting lunge  (`WW`), incremental speed lunge(`IS`).
    *   **Active Rules**: Strict Knee Angle (prevent over-extension), Arm-before-foot kinetic chain.
*   **Mode C: Free Bouting (對打)**
    *   **Target Actions**: All.
    *   **Active Rules**: Relaxed posture checks. Only trigger fatal errors (e.g., severe Guard Dropped). Focus on outputting tactical action sequences for the LLM.

---

## 2. Heuristic Mathematical Engine (姿勢判斷公式)

Implement a `HeuristicsEngine` class that receives a 30-frame window of YOLO coordinates and the `training_mode`. Note: In OpenCV/YOLO, the Y-axis increases downwards.

### 2.1 Base Math Utility
Implement a function to calculate the 2D angle $\theta$ between three points $A, B, C$ (where $B$ is the vertex):
$$ \theta = \arccos\left(\frac{\vec{BA} \cdot \vec{BC}}{|\vec{BA}| |\vec{BC}|}\right) \times \frac{180}{\pi} $$

### 2.2 Rule 1: Center of Mass Bounce (重心起伏)
*   **Trigger**: `Mode A`, when FenceNet predicts `SF` or `SB`.
*   **Logic**: Extract the Y-coordinate of the Pelvis (center of hips) across the 30 frames. Calculate the vertical variance:
    $$ \Delta Y = \max(Y_{pelvis}) - \min(Y_{pelvis}) $$
*   **Threshold**: If $\Delta Y > 0.05 \times \text{Bounding Box Height}$, return `Warning: Excessive vertical bouncing during footwork.`

### 2.3 Rule 2: Lunge Knee Over-extension (弓步膝蓋前傾)
*   **Trigger**: `Mode B`, when FenceNet predicts `R`, `JS`, or `WW`.
*   **Logic**: 
    1. Find the "Keyframe" where the lunge is fully extended (max X-distance between front ankle and back ankle).
    2. Calculate the angle of the front leg using joints: `Hip` (A), `Knee` (B), `Ankle` (C).
*   **Threshold**: If $\theta < 90^\circ$, return `Warning: Front knee angle is too acute. Over-lunging detected.`

### 2.4 Rule 3: Guard Dropped (持劍手掉落)
*   **Trigger**: All Modes (Tolerance varies: Mode A/B requires >10 frames, Mode C requires >20 frames).
*   **Logic**: Compare the Y-coordinate of the `front_wrist` and the `pelvis`.
*   **Threshold**: If $Y_{front\_wrist} > Y_{pelvis}$ (meaning the wrist is physically lower than the waist) for the specified number of consecutive frames, return `Warning: Weapon hand dropped below waist, exposing valid target.`

---

## 3. Validation UI & Burn-in Rendering (骨架分析UI呈現)

Build a rapid prototyping UI using `Gradio` to validate the pipeline synchronously. 

### 3.1 UI Layout
*   **Inputs**:
    *   `Video File Upload` (.mp4)
    *   `Radio Button`: Target Fencer (`Left` / `Right`)
    *   `Radio Button`: Training Mode (`Footwork` / `Target Practice` / `Free Bouting`)
    *   `Button`: Run Analysis
*   **Outputs**:
    *   `Video Player`: Displaying the rendered output video.
    *   `Data Table`: A drill-by-drill log (e.g., `Start Time | End Time | Action | Warning`).

### 3.2 OpenCV Burn-in Requirements (Frame-by-Frame Rendering)
To avoid front-end sync issues, the backend must use `cv2` to draw the analysis directly onto a new output `.mp4` file.
*   **Skeleton**: Draw the standard YOLO pose connections *only* for the targeted fencer.
*   **HUD (Heads-Up Display)**: In the top-left corner of the video, draw a semi-transparent black rectangle and use `cv2.putText` to display:
    *   `Mode`: [User Selected Mode]
    *   `State`: ACTIVE / IDLE (from Gatekeeper logic)
    *   `Action`: [FenceNet Prediction for current window]
    *   `Alert`: [Red text if a Heuristic Warning is triggered in this timeframe, otherwise empty]
*   **Visual Trigger**: If a heuristic warning is fired (e.g., Guard Dropped), draw a thick RED bounding box around the fencer for the duration of that action window. Otherwise, use a GREEN bounding box.

---

## 4. Agent Tasks

1.  **Update Inference Script**: Add the sliding window pipeline that integrates YOLO extraction, FenceNet prediction, and Heuristics dispatching based on `training_mode`.
2.  **Implement Math Utilities**: Code the vector angle math and the 3 heuristic rules defined above.
3.  **Build Gradio App**: Create `app.py` meeting the UI and OpenCV burn-in specifications.

# AI Fencing Coach MVP - Phase 3: LLM Integration & Data Persistence

**Objective**: Implement a local SQLite database to save user sessions and integrate an LLM API to generate a post-session coaching report based on heuristic logs.

## 1. Database Architecture (SQLite)
Create `database.py` to manage a local `fencing_coach.db`.
*   **Table `Users`**:
    *   `id` (PK), `name` (str), `handedness` (str: left/right), `height_cm` (int).
*   **Table `Sessions`**:
    *   `session_id` (PK), `user_id` (FK), `date` (timestamp), `training_mode` (str).
    *   `video_path` (str), `llm_summary` (text).
*   **Table `ActionLogs`**:
    *   `log_id` (PK), `session_id` (FK), `start_frame` (int), `action_label` (str).
    *   `heuristic_warning` (str - null if no error).

## 2. LLM Integration (Coach's Summary)
Implement `llm_agent.py` to call an LLM API (e.g., OpenAI or Google Gemini).
*   **Trigger**: This runs *after* the sliding window and heuristics engine have completely processed the video and populated the `ActionLogs` table for the current session.
*   **Prompt Construction**:
    1. Query the `ActionLogs` to aggregate stats (e.g., "Total actions: 12. Errors: 4 Guard Dropped, 1 Knee Over-extension").
    2. Inject User Profile (e.g., "User is a Right-handed fencer").
    3. **System Prompt Template**:
       ```text
       You are an expert fencing coach. Review the following session data for your student.
       Training Mode: {training_mode}
       Action Stats: {aggregated_stats}
       Write a short, encouraging, but highly specific technical paragraph (under 100 words) summarizing their performance and telling them exactly what biomechanical flaw to focus on fixing next. Do not list timecodes.
       ```
*   **Storage**: Save the resulting string to the `Sessions.llm_summary` field.

## 3. UI Updates (Gradio)
Update `app.py` to reflect the new features:
1.  **Sidebar / Top Header**: Add a basic User Selection (e.g., dropdown to select "User A" or "Create New User").
2.  **Output Layout**: 
    *   Top: The processed Video Player (with HUD).
    *   Middle: A Markdown text block displaying the `LLM Coach Summary`.
    *   Bottom: The detailed Drill-by-Drill Data Table.
3.  **History Tab**: Add a new tab in Gradio that queries the `Sessions` table and displays past LLM summaries and video links.