# Project Overview: AI Fencing Audio Coach (Native Desktop MVP)

## Objective
Build a Minimum Viable Product (MVP) for a real-time AI Fencing Coach. The system runs natively on a desktop/laptop using standard OpenCV and the built-in/USB webcam. It performs lightweight pose estimation on two fencers (side-view), evaluates heuristic rules based on dynamic distance and stance metrics, and triggers real-time audio feedback without interrupting the video feed.

## Architecture & Tech Stack
* **Framework:** Pure Python 3.10+
* **Real-time Video:** `opencv-python` using `cv2.VideoCapture(0)`
* **Computer Vision:** `ultralytics` (`yolov8n-pose.pt`) for multi-person tracking and keypoint extraction.
* **Audio Output:** `pyttsx3` (Offline Text-to-Speech) running asynchronously via `threading`.
* **UI:** Native `cv2.imshow` window with OpenCV drawing functions (COCO skeletons, bounding boxes, text labels, exponential moving average FPS counter).

## Target Audience & Scope
Beginner/Intermediate fencers. The CV model tracks lower-body heuristics (ankles, knees, hips) and bounding box dimensions to manage distance and footwork. Tactical blade tracking is out of scope for this MVP.

## Core Data Structures & State Management
* `fencer_L` / `fencer_R`: Detected keypoints and bounding boxes. (Left fencer ALWAYS has the smaller center X-coordinate).
* `avg_height`: The average bounding box height of `fencer_L` and `fencer_R` in pixels. Used as a dynamic scale reference to make the system camera-distance agnostic.
* `engagement_dist`: Absolute X-axis pixel distance between fencer_L's front ankle (max X) and fencer_R's front ankle (min X).
* `audio_cooldown`: Timestamp tracker (e.g., 4.0 seconds) to prevent spamming audio instructions.

## Feature 1: Video Pipeline & Pose Extraction
1. Initialize webcam capture. Create a `while True:` loop to read frames continuously.
2. Run YOLOv8 pose estimation on each frame (`verbose=False` to avoid terminal spam).
3. Extract bounding boxes and keypoints. Filter out background people by keeping ONLY the 2 persons with the highest confidence or largest bounding box areas.

## Feature 2: Robust Tracking & ID Assignment
Since fencing occurs on a linear 1D strip, fencers rarely swap sides.
1. Compare the X-coordinates of the center of the bounding boxes (or hips) of the 2 detected persons.
2. The person with the smaller X-coordinate is strictly assigned as `Fencer L`. The larger X-coordinate is `Fencer R`.

## Feature 3: Rule-Based Evaluation Engine (The "Brain")
Implement dynamic proportional thresholding to ensure the logic works regardless of how far the camera is from the fencers.

* **Rule 1: Distance Management (距離太近)**
  * *Global Config:* `DISTANCE_MULTIPLIER = 1.0`
  * *Dynamic Threshold Calculation:* `dynamic_threshold = avg_height * DISTANCE_MULTIPLIER`
  * *Condition:* If `engagement_dist < dynamic_threshold` AND `time.time() - last_audio_time > audio_cooldown`.
  * *Trigger Output:* Flash red text "TOO CLOSE!" on screen and send trigger string to Audio Dispatcher.

* **Rule 2: Footwork Form (腳步走太大) - [To Be Implemented]**
  * *Logic:* Evaluate `stance_width` (distance between a fencer's own front and back feet) against their own bounding box height.

## Feature 4: Async Audio Dispatcher
* Initialize `pyttsx3`.
* Create a non-blocking `speak_async(text)` function using `threading.Thread`.
* Receive the trigger string from the Rules Engine.
* This MUST run asynchronously so `pyttsx3.runAndWait()` never freezes the main `cv2` video inference loop.

## Visual Debugging UI
Draw the following directly onto the OpenCV frame:
1. YOLO skeletons and bounding boxes for both fencers.
2. Text labels above them: "Fencer L" and "Fencer R".
3. Top center text: `Dist: [engagement_dist] / Thresh: [dynamic_threshold]` (in cyan).
4. Top right text: Smoothed FPS counter.
5. Provide a clean exit: Press 'q' to release the camera (`cap.release()`) and destroy windows.