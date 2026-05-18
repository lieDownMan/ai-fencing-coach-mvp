import cv2
import numpy as np
import time
from collections import deque
import logging
import os
from PIL import Image, ImageDraw, ImageFont

from inference.sliding_window import SlidingWindowInference
from inference.activity_gatekeeper import ActivityGatekeeper
from inference.heuristics_engine import HeuristicsEngine
from inference.target_tracker import TargetTracker
from src.pose_estimation import PoseEstimator
from src.preprocessing import SpatialNormalizer
from src.realtime.feedback_scheduler import FeedbackDecision, FeedbackScheduler
from src.realtime.realtime_voice_coach import RealtimeVoiceCoach

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LiveCoach")

# ---------------------------------------------------------------------------
# Skeleton drawing constants
# ---------------------------------------------------------------------------
# COCO-style limb connections using joint names from PoseEstimator
SKELETON_LIMBS = [
    ("nose", "front_shoulder"),
    ("front_shoulder", "front_elbow"),
    ("front_elbow", "front_wrist"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
    ("front_shoulder", "right_hip"),
    ("front_shoulder", "left_hip"),
]
JOINT_COLOR = (0, 255, 255)   # yellow (BGR)
LIMB_COLOR = (255, 200, 100)  # light blue (BGR)

# Load a CJK-capable font for text overlay
try:
    # macOS system font that supports Chinese
    _FONT = ImageFont.truetype("/System/Library/Fonts/PingFang.ttc", 42)
    _FONT_SMALL = ImageFont.truetype("/System/Library/Fonts/PingFang.ttc", 30)
    _FONT_WARNING = ImageFont.truetype("/System/Library/Fonts/PingFang.ttc", 48)
except Exception:
    _FONT = ImageFont.load_default()
    _FONT_SMALL = _FONT
    _FONT_WARNING = _FONT

def _draw_skeleton(frame: np.ndarray, skeleton: dict) -> np.ndarray:
    """Draw skeleton joints and limbs on the frame."""
    if not skeleton:
        return frame

    # Labels for key joints (Chinese)
    JOINT_LABELS = {
        "front_wrist":    ("前手", (0, 255, 0)),     # green = front/sword hand
        "front_elbow":    ("前肘", (0, 255, 0)),
        "front_shoulder": ("前肩", (0, 255, 0)),
        "nose":           ("鼻",  (255, 255, 255)),
        "left_hip":       ("左髖", (200, 200, 200)),
        "right_hip":      ("右髖", (200, 200, 200)),
        "left_knee":      ("左膝", (200, 200, 200)),
        "right_knee":     ("右膝", (200, 200, 200)),
        "left_ankle":     ("左踝", (200, 200, 200)),
        "right_ankle":    ("右踝", (200, 200, 200)),
    }

    # Draw limbs
    for j1, j2 in SKELETON_LIMBS:
        p1 = skeleton.get(j1)
        p2 = skeleton.get(j2)
        if p1 is not None and p2 is not None:
            pt1 = (int(p1[0]), int(p1[1]))
            pt2 = (int(p2[0]), int(p2[1]))
            # Front arm limbs in green, others in light blue
            is_front = j1.startswith("front") or j2.startswith("front")
            color = (0, 255, 0) if is_front else LIMB_COLOR
            cv2.line(frame, pt1, pt2, color, 2, cv2.LINE_AA)

    # Draw joints with labels
    for joint_name, coords in skeleton.items():
        pt = (int(coords[0]), int(coords[1]))
        label_info = JOINT_LABELS.get(joint_name)
        if label_info:
            label_text, label_color = label_info
            # Larger dot for front hand joints
            radius = 7 if joint_name.startswith("front") else 4
            cv2.circle(frame, pt, radius, label_color, -1, cv2.LINE_AA)
            # Draw label text next to joint
            cv2.putText(frame, label_text, (pt[0] + 10, pt[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_color, 1, cv2.LINE_AA)
        else:
            cv2.circle(frame, pt, 4, JOINT_COLOR, -1, cv2.LINE_AA)

    return frame


def _put_text_pil(
    frame: np.ndarray,
    text: str,
    position: tuple,
    color: tuple = (255, 255, 255),
    font=None,
) -> np.ndarray:
    """Render text with PIL (supports Chinese) and composite onto the OpenCV frame."""
    if font is None:
        font = _FONT
    # Convert BGR → RGB for PIL
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)
    # Draw shadow for readability
    sx, sy = position
    draw.text((sx + 2, sy + 2), text, font=font, fill=(0, 0, 0))
    draw.text((sx, sy), text, font=font, fill=color)
    # Convert RGB → BGR back to OpenCV
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

class LiveVideoPipeline:
    def __init__(
        self,
        target_side="left",
        training_mode="Free Bouting",
        pose_model=None,
        pose_backend="ultralytics",
        voice_enabled=True,
        focus_errors=None,
        mute_errors=None,
        only_errors=None,
        only_selected=False,
    ):
        self.target_side = target_side
        self.training_mode = training_mode
        self.pose_estimator = PoseEstimator(model_path=pose_model, backend=pose_backend)
        self.target_tracker = TargetTracker(target_side=target_side)
        self.gatekeeper = ActivityGatekeeper(fps=30)
        self.sliding_window = SlidingWindowInference(model_path="weights/fencenet/best_model.pth", device="auto")
        self.heuristics = HeuristicsEngine(target_side=target_side, training_mode=training_mode)
        self.normalizer = SpatialNormalizer()
        
        self.voice_coach = None
        if voice_enabled:
            try:
                self.voice_coach = RealtimeVoiceCoach()
                logger.info("Voice Coach initialized.")
            except Exception as e:
                logger.error(f"Voice Coach failed to initialize: {e}")
        else:
            logger.info("Voice Coach disabled.")

        self.window_size = self.sliding_window.window_size
        self.stride = self.sliding_window.stride
        
        self.raw_skeletons = deque(maxlen=self.window_size)
        self.normalized_skeletons = deque(maxlen=self.window_size)
        
        self.frame_idx = 0
        self.last_inference_idx = 0
        self.current_action = "Idle"
        self.current_warnings = []       # list of warning strings to display
        self.current_feedback_items = [] # ranked FeedbackItem objects for web/UI status
        self.warning_frames_left = 0
        self.prev_gatekeeper_active = False  # Track transitions for normalizer re-fit
        playbook = self.voice_coach.playbook if self.voice_coach else None
        self.feedback_scheduler = FeedbackScheduler(
            playbook=playbook,
            focus_errors=focus_errors,
            mute_errors=mute_errors,
            only_errors=only_errors,
            only_selected=only_selected,
        )
        
    def _pixel_to_xyn(self, skeleton: dict, frame_h: int, frame_w: int) -> dict:
        """Convert pixel (xy) skeleton to normalized (xyn) skeleton (0.0~1.0).

        Training data (convert_to_json.py) uses keypoints.xyn, so we must
        feed the same coordinate space to the SpatialNormalizer / model.
        """
        xyn = {}
        for joint_name, coords in skeleton.items():
            xyn[joint_name] = (coords[0] / frame_w, coords[1] / frame_h)
        return xyn

    def _apply_feedback_decision(self, decision: FeedbackDecision) -> None:
        """Apply one scheduler decision to voice output and visible warnings."""
        self.current_feedback_items = decision.visual_items
        self.current_warnings = [item.message for item in decision.visual_items]
        self.warning_frames_left = 90 if self.current_warnings else 0

        if not decision.voice_error_key:
            return

        cue = decision.voice_message or decision.voice_error_key
        print(f"\n[COACHING FEEDBACK] {cue}")
        if self.voice_coach:
            self.voice_coach.speak_async(decision.voice_error_key)

    def process_frame(self, frame, draw_hud: bool = True):
        height, width = frame.shape[:2]
        
        # 1. Pose Estimation & Tracking
        detections = self.pose_estimator.extract_frame_fencers(frame, persist_track=True)
        target_skel, opp_skel = self.target_tracker.process_frame_detections(detections, self.frame_idx)
        
        # Draw skeleton overlay (uses pixel coords)
        if target_skel:
            frame = _draw_skeleton(frame, target_skel)
        if opp_skel:
            frame = _draw_skeleton(frame, opp_skel)
        
        # 2. Activity Gating (uses pixel coords for distance/angle checks)
        is_active = self.gatekeeper.update(target_skel, opp_skel, width, self.target_side)
        
        # Re-fit normalizer on IDLE→ACTIVE transitions to prevent coordinate drift
        if is_active and not self.prev_gatekeeper_active and target_skel:
            self.normalizer.reference_nose = None  # Reset to re-fit
            self.normalizer.scale_factor = None
        self.prev_gatekeeper_active = is_active
        
        # Debug: log pipeline state every 30 frames (~1s)
        if self.frame_idx % 30 == 0:
            has_skel = target_skel is not None
            buf_fill = len(self.normalized_skeletons)
            print(f"[DEBUG] frame={self.frame_idx} skel={has_skel} state={self.gatekeeper.state} "
                  f"active={is_active} buf={buf_fill}/{self.window_size} action={self.current_action}")
        
        # 3. Normalization & Buffer Management
        if target_skel:
            self.raw_skeletons.append(target_skel)
            
            # Convert pixel coords → xyn (0~1) to match training data format
            xyn_skel = self._pixel_to_xyn(target_skel, height, width)
            
            if is_active:
                try:
                    if self.normalizer.reference_nose is None:
                        self.normalizer.fit([xyn_skel])
                    norm_dict = self.normalizer.normalize_skeleton(xyn_skel)
                    norm_arr = np.array([norm_dict[j] for j in self.normalizer.MODEL_JOINT_NAMES])
                except Exception:
                    norm_arr = np.zeros((9, 2))
            else:
                norm_arr = np.zeros((9, 2))
                
            self.normalized_skeletons.append(norm_arr)
        else:
            self.raw_skeletons.append({})
            self.normalized_skeletons.append(np.zeros((9, 2)))

        # 4. Inference & Heuristics when buffer is full and stride is met
        if len(self.normalized_skeletons) == self.window_size and (self.frame_idx - self.last_inference_idx) >= self.stride:
            skel_array = np.array(self.normalized_skeletons)
            
            # Real-time: classify the single window directly (skip NMS which filters out Idle)
            window_results = self.sliding_window._classify_windows(skel_array)
            
            if window_results:
                best_result = max(window_results, key=lambda x: x["confidence"])
                self.current_action = best_result["action"]
                conf = best_result["confidence"]
                print(f"[INFERENCE] action={self.current_action} conf={conf:.3f}")
                
                posture_errors = []
                if self.current_action != "Idle":
                    posture_errors = self.heuristics.evaluate(
                        [best_result],
                        list(self.raw_skeletons),
                    )

                active_error_keys = [
                    err.get("error_key", "")
                    for err in posture_errors
                    if err.get("error_key")
                ]
                decision = self.feedback_scheduler.update(
                    active_error_keys,
                    now=time.monotonic(),
                )
                self._apply_feedback_decision(decision)
            self.last_inference_idx = self.frame_idx
            
            
        if draw_hud:
            # Draw HUD text using PIL (supports Chinese characters)
            frame = _put_text_pil(frame, f"State: {self.gatekeeper.state}", (20, 10), color=(0, 255, 0), font=_FONT_SMALL)
            frame = _put_text_pil(frame, f"Action: {self.current_action}", (20, 50), color=(0, 165, 255), font=_FONT)
            
            if self.warning_frames_left > 0:
                y_offset = 105
                for index, warn_text in enumerate(self.current_warnings):
                    prefix = "PRIMARY" if index == 0 else "NEXT"
                    font = _FONT_WARNING if index == 0 else _FONT_SMALL
                    color = (255, 80, 80) if index == 0 else (255, 200, 80)
                    frame = _put_text_pil(frame, f"{prefix}: {warn_text}", (20, y_offset), color=color, font=font)
                    y_offset += 60 if index == 0 else 42  # spacing between lines
                self.warning_frames_left -= 1
        else:
            # Still decrement the warning timer if HUD is disabled so web UI can sync
            if self.warning_frames_left > 0:
                self.warning_frames_left -= 1
                
        self.frame_idx += 1
        return frame


if __name__ == "__main__":
    import argparse

    def _split_error_keys(value):
        if not value:
            return []
        return [item.strip() for item in value.split(",") if item.strip()]

    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0", help="Camera source index (0) or video path")
    parser.add_argument("--mode", default="Free Bouting", choices=["Footwork", "Target Practice", "Free Bouting"])
    parser.add_argument("--target-side", default="left", choices=["left", "right"], help="Which fencer to coach in the camera frame")
    parser.add_argument("--pose-model", default=None, help="Path/name for YOLO pose weights (default: yolov8n-pose.pt)")
    parser.add_argument("--pose-backend", default="ultralytics", choices=["ultralytics", "mock"], help="Use mock for smoke checks without pose inference")
    parser.add_argument("--no-voice", action="store_true", help="Disable offline spoken cues")
    parser.add_argument("--focus-errors", default="", help="Comma-separated error keys to prioritize, e.g. stance_too_high,bounce_excessive")
    parser.add_argument("--mute-errors", default="", help="Comma-separated error keys to hide/silence")
    parser.add_argument("--only-errors", default="", help="Comma-separated error keys to exclusively show/speak")
    args = parser.parse_args()

    # Determine if source is integer (webcam) or string (file)
    try:
        source = int(args.source)
    except ValueError:
        source = args.source

    if isinstance(source, int) and os.name == "nt":
        cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Cannot open video source {source}")
        exit()
        
    pipeline = LiveVideoPipeline(
        target_side=args.target_side,
        training_mode=args.mode,
        pose_model=args.pose_model,
        pose_backend=args.pose_backend,
        voice_enabled=not args.no_voice,
        focus_errors=_split_error_keys(args.focus_errors),
        mute_errors=_split_error_keys(args.mute_errors),
        only_errors=_split_error_keys(args.only_errors),
    )
    print("=======================================")
    print(" Live AI Fencing Coach Started!")
    print(f" Source: {source} | Mode: {args.mode} | Target: {args.target_side}")
    if args.focus_errors:
        print(f" Focus errors: {args.focus_errors}")
    if args.mute_errors:
        print(f" Muted errors: {args.mute_errors}")
    if args.only_errors:
        print(f" Only errors: {args.only_errors}")
    print(" Press 'q' to quit.")
    print("=======================================")
    
    drop_count = 0
    MAX_DROPS = 30  # Allow up to 30 consecutive dropped frames (~1 second at 30fps)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            drop_count += 1
            if drop_count >= MAX_DROPS:
                print(f"\n[ERROR] Camera lost: {MAX_DROPS} consecutive frames dropped. Exiting.")
                break
            print(f"[WARN] Frame dropped ({drop_count}/{MAX_DROPS}), retrying...")
            cv2.waitKey(33)  # Wait ~1 frame before retrying
            continue
        drop_count = 0  # Reset on successful read
            
        # Process the frame
        out_frame = pipeline.process_frame(frame)
        
        # Display
        cv2.imshow("Live Fencing Coach", out_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    print("\n[INFO] Shutting down...")
    cap.release()
    cv2.destroyAllWindows()
    if pipeline.voice_coach:
        pipeline.voice_coach.shutdown()
