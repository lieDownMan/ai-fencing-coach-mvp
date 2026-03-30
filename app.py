"""
AI Fencing Coach MVP — Pure OpenCV + YOLOv8n-pose
Dual-mode: live webcam OR imported video file.
Press 'q' to quit.

Usage:
  python app.py                    # webcam mode (default)
  python app.py --video path.mp4   # video file mode → outputs processed file
"""

import sys
import os
import cv2
import time
import argparse
import threading
import pyttsx3
import numpy as np
from ultralytics import YOLO

# ── Configuration ────────────────────────────────────────────
MODEL_PATH = "yolov8n-pose.pt"          # auto-downloads on first run
YOLO_CONF = 0.5                          # detection confidence threshold
KP_CONF = 0.4                           # keypoint confidence threshold
WINDOW_NAME = "AI Fencing Coach"

# Audio Rule thresholds
DISTANCE_MULTIPLIER = 1.0               # "too close" when dist < multiplier × avg fencer height
AUDIO_COOLDOWN = 4.0                    # seconds between audio alerts

# ── TTS Setup ────────────────────────────────────────────────
def speak_async(text: str):
    """Run TTS in a background thread so we do not block the video loop."""
    def _speak():
        try:
            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            print(f"[TTS Error] {e}")

    threading.Thread(target=_speak, daemon=True).start()

# ── COCO-pose skeleton (17 keypoints) ────────────────────────
# 0:nose 1:L_eye 2:R_eye 3:L_ear 4:R_ear
# 5:L_shoulder 6:R_shoulder 7:L_elbow 8:R_elbow 9:L_wrist 10:R_wrist
# 11:L_hip 12:R_hip 13:L_knee 14:R_knee 15:L_ankle 16:R_ankle
SKELETON_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),        # head
    (5, 6),                                  # shoulders
    (5, 7), (7, 9),                          # left arm
    (6, 8), (8, 10),                         # right arm
    (5, 11), (6, 12), (11, 12),              # torso
    (11, 13), (13, 15),                      # left leg
    (12, 14), (14, 16),                      # right leg
]

# Colours (BGR)
COLOR_L = (0, 255, 128)       # green  → Fencer L
COLOR_R = (0, 128, 255)       # orange → Fencer R
COLOR_SINGLE = (255, 200, 0)  # cyan-ish for single detection
COLOR_WARN = (0, 0, 255)      # red for warning text


# ── Drawing helpers ──────────────────────────────────────────
def draw_skeleton(img, kps, color, thickness=2):
    """Draw keypoint dots and skeleton edges."""
    for i, j in SKELETON_EDGES:
        if kps[i][2] > KP_CONF and kps[j][2] > KP_CONF:
            pt1 = (int(kps[i][0]), int(kps[i][1]))
            pt2 = (int(kps[j][0]), int(kps[j][1]))
            cv2.line(img, pt1, pt2, color, thickness, cv2.LINE_AA)
    for x, y, c in kps:
        if c > KP_CONF:
            cv2.circle(img, (int(x), int(y)), 4, color, -1, cv2.LINE_AA)


def draw_bbox_label(img, bbox, label, color, thickness=2):
    """Draw bounding box with a filled label tag above it."""
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, txt_thick = 0.7, 2
    (tw, th), baseline = cv2.getTextSize(label, font, scale, txt_thick)
    tag_top = max(y1 - th - baseline - 8, 0)
    cv2.rectangle(img, (x1, tag_top), (x1 + tw + 8, y1), color, -1)
    cv2.putText(img, label, (x1 + 4, y1 - baseline - 2),
                font, scale, (0, 0, 0), txt_thick, cv2.LINE_AA)


def draw_fps(img, fps):
    """FPS counter in the top-right corner."""
    text = f"FPS: {fps:.0f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thick = 0.7, 2
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    x = img.shape[1] - tw - 12
    cv2.putText(img, text, (x, 28), font, scale, (0, 255, 0), thick, cv2.LINE_AA)


def draw_mode_badge(img, mode_text):
    """Mode indicator badge in the top-left corner."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thick = 0.6, 2
    (tw, th), baseline = cv2.getTextSize(mode_text, font, scale, thick)
    cv2.rectangle(img, (8, 6), (18 + tw, 14 + th + baseline), (60, 60, 60), -1)
    cv2.putText(img, mode_text, (12, 10 + th), font, scale, (200, 200, 200), thick, cv2.LINE_AA)


# ── Frame processing (shared by both modes) ─────────────────
def process_frame(frame, model, last_audio_time, enable_audio=True):
    """
    Run YOLO, assign fencer IDs, compute metrics, draw overlays,
    trigger audio. Returns (annotated_frame, last_audio_time).
    """
    results = model.predict(frame, verbose=False, conf=YOLO_CONF)
    result = results[0]

    engagement_dist = None
    dynamic_threshold = None

    if result.boxes is not None and len(result.boxes) > 0:
        boxes = result.boxes.xyxy.cpu().numpy()       # (N, 4)
        confs = result.boxes.conf.cpu().numpy()        # (N,)
        kps   = result.keypoints.data.cpu().numpy()    # (N, 17, 3)

        # ── Keep top-2 largest bounding boxes ─────────────
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        if len(areas) > 2:
            top2 = np.argsort(areas)[-2:]
        else:
            top2 = np.arange(len(areas))

        sel_boxes = boxes[top2]
        sel_kps   = kps[top2]

        # ── Sort by center-X → assign L / R ──────────────
        cx = (sel_boxes[:, 0] + sel_boxes[:, 2]) / 2.0
        order = np.argsort(cx)
        sel_boxes = sel_boxes[order]
        sel_kps   = sel_kps[order]

        if len(sel_boxes) == 2:
            labels = ["Fencer L", "Fencer R"]
            colors = [COLOR_L, COLOR_R]

            # ── Calculate Metrics ─────────────────────────
            bbox_L, bbox_R = sel_boxes[0], sel_boxes[1]
            kps_L, kps_R   = sel_kps[0], sel_kps[1]

            # Dynamic threshold from bounding-box heights
            height_L = bbox_L[3] - bbox_L[1]
            height_R = bbox_R[3] - bbox_R[1]
            avg_height = (height_L + height_R) / 2.0
            dynamic_threshold = avg_height * DISTANCE_MULTIPLIER

            # Ankle indices: 15 (L_ankle), 16 (R_ankle)
            # Front ankle for Fencer L = max X, Fencer R = min X
            if kps_L[15][2] > KP_CONF and kps_L[16][2] > KP_CONF:
                ankle_L_x = max(kps_L[15][0], kps_L[16][0])
            elif kps_L[15][2] > KP_CONF:
                ankle_L_x = kps_L[15][0]
            elif kps_L[16][2] > KP_CONF:
                ankle_L_x = kps_L[16][0]
            else:
                ankle_L_x = None

            if kps_R[15][2] > KP_CONF and kps_R[16][2] > KP_CONF:
                ankle_R_x = min(kps_R[15][0], kps_R[16][0])
            elif kps_R[15][2] > KP_CONF:
                ankle_R_x = kps_R[15][0]
            elif kps_R[16][2] > KP_CONF:
                ankle_R_x = kps_R[16][0]
            else:
                ankle_R_x = None

            if ankle_L_x is not None and ankle_R_x is not None:
                engagement_dist = abs(ankle_R_x - ankle_L_x)

        else:
            labels = ["Fencer"]
            colors = [COLOR_SINGLE]

        # ── Draw Skeletons & BBoxes ───────────────────────
        for bbox, keypoints, label, color in zip(sel_boxes, sel_kps, labels, colors):
            draw_bbox_label(frame, bbox, label, color)
            draw_skeleton(frame, keypoints, color)

    # ── Rules Engine & UI Drawing ─────────────────────────
    if engagement_dist is not None and dynamic_threshold is not None:
        # Draw distance + threshold at top center
        dist_text = f"Dist: {int(engagement_dist)} / Thresh: {int(dynamic_threshold)}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        dw, dh = cv2.getTextSize(dist_text, font, 1.0, 3)[0]
        cv2.putText(frame, dist_text, ((frame.shape[1] - dw) // 2, 40),
                    font, 1.0, (255, 255, 0), 3, cv2.LINE_AA)

        # Rule 1: Distance too close (dynamic threshold)
        if engagement_dist < dynamic_threshold:
            # Flash Warning Text
            warn_text = "TOO CLOSE!"
            ww, wh = cv2.getTextSize(warn_text, font, 1.5, 4)[0]
            cv2.putText(frame, warn_text, ((frame.shape[1] - ww) // 2, 90),
                        font, 1.5, COLOR_WARN, 4, cv2.LINE_AA)

            # Trigger Audio Dispatcher
            if enable_audio:
                current_time = time.time()
                if current_time - last_audio_time > AUDIO_COOLDOWN:
                    last_audio_time = current_time
                    speak_async("Warning, distance too close. Watch your steps.")

    return frame, last_audio_time


# ── Webcam mode ──────────────────────────────────────────────
def run_webcam(model):
    """Live webcam capture + real-time processing."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("[INFO] Webcam opened. Press 'q' to quit.")
    fps = 0.0
    prev_tick = cv2.getTickCount()
    last_audio_time = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Empty frame, skipping.")
            continue

        frame, last_audio_time = process_frame(frame, model, last_audio_time)

        # FPS
        curr_tick = cv2.getTickCount()
        elapsed = (curr_tick - prev_tick) / cv2.getTickFrequency()
        prev_tick = curr_tick
        fps = 0.9 * fps + 0.1 * (1.0 / max(elapsed, 1e-6))
        draw_fps(frame, fps)
        draw_mode_badge(frame, "LIVE")

        cv2.imshow(WINDOW_NAME, frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Exited cleanly.")


# ── Video file mode ──────────────────────────────────────────
def run_video(model, video_path):
    """Process a video file and write annotated output."""
    if not os.path.isfile(video_path):
        print(f"[ERROR] Video file not found: {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return

    # Read video properties
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Build output path: input.mp4 → input_processed.mp4
    base, ext = os.path.splitext(video_path)
    output_path = f"{base}_processed{ext}"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps_in, (w, h))

    print(f"[INFO] Processing video: {video_path}")
    print(f"[INFO] Resolution: {w}x{h} | FPS: {fps_in:.1f} | Frames: {total_frames}")
    print(f"[INFO] Output will be saved to: {output_path}")
    print("[INFO] Press 'q' to cancel early.")

    last_audio_time = 0.0
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        frame, last_audio_time = process_frame(
            frame, model, last_audio_time, enable_audio=False
        )

        # Progress badge instead of FPS
        progress = f"Frame {frame_idx}/{total_frames} ({100*frame_idx/max(total_frames,1):.0f}%)"
        draw_mode_badge(frame, f"VIDEO | {progress}")

        writer.write(frame)

        # Preview window (press q to cancel)
        cv2.imshow(WINDOW_NAME, frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("[INFO] Cancelled by user.")
            break

    writer.release()
    cap.release()
    cv2.destroyAllWindows()
    print(f"[INFO] Done! Processed video saved to: {output_path}")


# ── Entry point ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="AI Fencing Coach MVP")
    parser.add_argument(
        "--video", type=str, default=None,
        help="Path to a video file. If omitted, the webcam is used."
    )
    args = parser.parse_args()

    print(f"[INFO] Loading {MODEL_PATH} …")
    model = YOLO(MODEL_PATH)

    if args.video:
        run_video(model, args.video)
    else:
        run_webcam(model)


if __name__ == "__main__":
    main()
