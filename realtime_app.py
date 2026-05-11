import cv2
import numpy as np
from collections import deque
import logging
import os

from inference.sliding_window import SlidingWindowInference
from inference.activity_gatekeeper import ActivityGatekeeper
from inference.heuristics_engine import HeuristicsEngine
from inference.target_tracker import TargetTracker
from src.pose_estimation import PoseEstimator
from src.preprocessing import SpatialNormalizer
from realtime_voice_coach import RealtimeVoiceCoach

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LiveCoach")

class LiveVideoPipeline:
    def __init__(
        self,
        target_side="left",
        training_mode="Free Bouting",
        pose_model=None,
        pose_backend="ultralytics",
        voice_enabled=True,
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
        self.current_warning = ""
        self.warning_frames_left = 0
        
    def process_frame(self, frame):
        width = frame.shape[1]
        
        # 1. Pose Estimation & Tracking
        detections = self.pose_estimator.extract_frame_fencers(frame, persist_track=True)
        target_skel, opp_skel = self.target_tracker.process_frame_detections(detections, self.frame_idx)
        
        # 2. Activity Gating
        is_active = self.gatekeeper.update(target_skel, opp_skel, width, self.target_side)
        
        # 3. Normalization & Buffer Management
        if target_skel:
            self.raw_skeletons.append(target_skel)
            
            if is_active:
                try:
                    if self.normalizer.reference_nose is None:
                        self.normalizer.fit([target_skel])
                    norm_dict = self.normalizer.normalize_skeleton(target_skel)
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
            
            # Predict action for the current window
            action_segments_raw = self.sliding_window.run(skel_array)
            
            if action_segments_raw:
                best_segment = max(action_segments_raw, key=lambda x: x["confidence"])
                self.current_action = best_segment["action"]
                
                # Check Heuristics
                posture_errors = self.heuristics.evaluate(
                    [best_segment], 
                    list(self.raw_skeletons)
                )
                
                for err in posture_errors:
                    error_key = err.get("error_key", "")
                    if error_key:
                        cue = error_key
                        if self.voice_coach:
                            cue = self.voice_coach.get_cue(error_key) or error_key
                            print(f"\n[COACHING FEEDBACK] {cue}")
                            self.voice_coach.speak_async(error_key)
                        else:
                            print(f"\n[COACHING FEEDBACK] {error_key}")
                            
                        self.current_warning = cue
                        self.warning_frames_left = 60 # Show warning for ~2 seconds
                        break # Only trigger one voice cue at a time
            self.last_inference_idx = self.frame_idx
            
        # Draw on frame
        cv2.putText(frame, f"State: {self.gatekeeper.state}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Action: {self.current_action}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 165, 0), 2)
        
        if self.warning_frames_left > 0:
            cv2.putText(frame, f"WARNING: {self.current_warning}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            self.warning_frames_left -= 1
            
        self.frame_idx += 1
        return frame


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0", help="Camera source index (0) or video path")
    parser.add_argument("--mode", default="Free Bouting", choices=["Footwork", "Target Practice", "Free Bouting"])
    parser.add_argument("--target-side", default="left", choices=["left", "right"], help="Which fencer to coach in the camera frame")
    parser.add_argument("--pose-model", default=None, help="Path/name for YOLO pose weights (default: yolov8n-pose.pt)")
    parser.add_argument("--pose-backend", default="ultralytics", choices=["ultralytics", "mock"], help="Use mock for smoke checks without pose inference")
    parser.add_argument("--no-voice", action="store_true", help="Disable offline spoken cues")
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
    )
    print("=======================================")
    print(" Live AI Fencing Coach Started!")
    print(f" Source: {source} | Mode: {args.mode} | Target: {args.target_side}")
    print(" Press 'q' to quit.")
    print("=======================================")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Process the frame
        out_frame = pipeline.process_frame(frame)
        
        # Display
        cv2.imshow("Live Fencing Coach", out_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
