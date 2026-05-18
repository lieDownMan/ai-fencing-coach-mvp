"""
Sliding Window Inference — Action Spotter for continuous videos.

Slides a fixed-size window across a normalized skeleton timeline,
classifies each window with FenceNetV2, then applies Non-Maximum
Suppression (NMS) to merge overlapping detections.

Spec reference: fixing_app.md § Module 1
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from src.models.fencenet_v2 import FenceNetV2
from src.data.fencing_dataset import CLASS_NAMES, NUM_CHANNELS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults (aligned with trained model and spec)
# ---------------------------------------------------------------------------
WINDOW_SIZE = 28          # must match trained model input time-steps
STRIDE = 10              # ~0.33 s at 30 fps (spec value)
CONFIDENCE_THRESHOLD = 0.6
ATTACKING_ACTIONS = {"R", "IS", "WW", "JS"}


class SlidingWindowInference:
    """
    Slide a fixed-size window over a skeleton timeline, classify each window
    with FenceNetV2, and apply NMS to produce a clean action timeline.

    Parameters
    ----------
    model_path : str or Path, optional
        Path to ``best_model.pth``.  If *None* the model runs with random
        weights (useful for smoke-testing the pipeline).
    device : str
        ``"cuda"`` or ``"cpu"``.
    window_size : int
        Number of frames per window (must match model training).
    stride : int
        Step size between consecutive windows.
    confidence_threshold : float
        Windows below this softmax confidence are labelled ``Idle``.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cpu",
        window_size: int = WINDOW_SIZE,
        stride: int = STRIDE,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
    ):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.window_size = window_size
        self.stride = stride
        self.confidence_threshold = confidence_threshold
        self.class_names = list(CLASS_NAMES)

        # Build model
        self.model = FenceNetV2(input_channels=NUM_CHANNELS)
        if model_path is not None:
            self._load_weights(model_path)
        self.model.to(self.device)
        self.model.eval()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        skeleton_array: np.ndarray,
    ) -> List[Dict[str, Any]]:
        """
        Run sliding-window inference + NMS on a full skeleton timeline.

        Parameters
        ----------
        skeleton_array : np.ndarray
            Shape ``(T, 9, 2)`` — spatially normalized skeleton sequence.

        Returns
        -------
        list[dict]
            Merged action segments, each containing::

                {"start_frame": int, "end_frame": int,
                 "action": str, "confidence": float}
        """
        raw_windows = self._classify_windows(skeleton_array)
        merged = self._nms(raw_windows)
        return merged

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _classify_windows(
        self, skeleton_array: np.ndarray
    ) -> List[Dict[str, Any]]:
        """Produce one prediction per sliding window."""
        T = skeleton_array.shape[0]
        results: List[Dict[str, Any]] = []

        if T < self.window_size:
            logger.warning(
                "Video has %d frames, less than window_size=%d — skipping.",
                T,
                self.window_size,
            )
            return results

        # Build all windows
        starts = list(range(0, T - self.window_size + 1, self.stride))
        if not starts:
            return results

        windows = np.stack(
            [skeleton_array[s : s + self.window_size] for s in starts],
            axis=0,
        )  # (N, window_size, 9, 2)

        # Reshape: (N, ws, 9, 2) → (N, ws, 18) → (N, 18, ws)
        N = windows.shape[0]
        flat = windows.reshape(N, self.window_size, -1)      # (N, ws, 18)
        tensor = (
            torch.from_numpy(flat)
            .float()
            .permute(0, 2, 1)                                 # (N, 18, ws)
            .to(self.device)
        )

        # Batch inference
        with torch.no_grad():
            logits = self.model(tensor)                       # (N, 6)
            probs = F.softmax(logits, dim=1)                  # (N, 6)
            confs, preds = probs.max(dim=1)                   # (N,), (N,)

        confs = confs.cpu().numpy()
        preds = preds.cpu().numpy()

        for i, start in enumerate(starts):
            pred_idx = int(preds[i])
            conf = float(confs[i])
            action = self.class_names[pred_idx] if conf >= self.confidence_threshold else "Idle"
            results.append(
                {
                    "start_frame": start,
                    "end_frame": start + self.window_size,
                    "action": action,
                    "confidence": conf,
                    "class_idx": pred_idx,
                }
            )

        return results

    def _nms(self, windows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Non-Maximum Suppression: merge consecutive overlapping windows
        with the same *attacking* action, keeping only the highest
        confidence occurrence.

        Non-attacking actions (SF, SB) and Idle are kept individually
        (but Idle is filtered out of final output).
        """
        if not windows:
            return []

        merged: List[Dict[str, Any]] = []
        current_group: List[Dict[str, Any]] = [windows[0]]

        for win in windows[1:]:
            prev = current_group[-1]
            same_action = win["action"] == prev["action"]
            is_attacking = win["action"] in ATTACKING_ACTIONS
            overlapping = win["start_frame"] < prev["end_frame"]

            if same_action and is_attacking and overlapping:
                # Extend the current group
                current_group.append(win)
            else:
                # Flush current group
                merged.append(self._best_of_group(current_group))
                current_group = [win]

        # Flush last group
        merged.append(self._best_of_group(current_group))

        # Remove Idle segments from final output
        return [seg for seg in merged if seg["action"] != "Idle"]

    @staticmethod
    def _best_of_group(group: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Return the window with the highest confidence, spanning the full group range."""
        best = max(group, key=lambda w: w["confidence"])
        return {
            "start_frame": group[0]["start_frame"],
            "end_frame": group[-1]["end_frame"],
            "action": best["action"],
            "confidence": best["confidence"],
        }

    def _load_weights(self, model_path: str) -> None:
        """Load trained weights from ``best_model.pth``."""
        path = Path(model_path)
        if not path.exists():
            logger.warning("Model weights not found: %s — using random weights.", path)
            return

        checkpoint = torch.load(str(path), map_location=self.device, weights_only=True)

        # Support both plain state_dict and wrapped checkpoint formats
        if isinstance(checkpoint, dict):
            if "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            elif "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            elif self._looks_like_state_dict(checkpoint):
                state_dict = checkpoint
            else:
                logger.warning("Unrecognized checkpoint format — using random weights.")
                return
        else:
            logger.warning("Checkpoint is not a dict — using random weights.")
            return

        self.model.load_state_dict(state_dict)
        logger.info("Loaded FenceNetV2 weights from %s", path)

    @staticmethod
    def _looks_like_state_dict(d: dict) -> bool:
        return bool(d) and all(hasattr(v, "shape") for v in d.values())

import cv2
from typing import Dict, Any, List
import numpy as np

from .activity_gatekeeper import ActivityGatekeeper
from .heuristics_engine import HeuristicsEngine
from .target_tracker import TargetTracker
from src.pose_estimation import PoseEstimator
from src.preprocessing import SpatialNormalizer

def scale_pose_detections(
    detections: List[Dict[str, Any]],
    scale_x: float,
    scale_y: float,
) -> List[Dict[str, Any]]:
    """Scale pose detection geometry back to the original frame size."""
    if abs(scale_x - 1.0) < 1e-6 and abs(scale_y - 1.0) < 1e-6:
        return detections

    scaled = []
    for detection in detections:
        item = dict(detection)
        if "bbox" in item and item["bbox"] is not None:
            x1, y1, x2, y2 = item["bbox"]
            item["bbox"] = [
                float(x1) * scale_x,
                float(y1) * scale_y,
                float(x2) * scale_x,
                float(y2) * scale_y,
            ]
        if "center" in item and item["center"] is not None:
            cx, cy = item["center"]
            item["center"] = [float(cx) * scale_x, float(cy) * scale_y]
        if "area" in item:
            item["area"] = float(item["area"]) * scale_x * scale_y
        if "skeleton" in item and item["skeleton"] is not None:
            item["skeleton"] = {
                joint: (float(point[0]) * scale_x, float(point[1]) * scale_y)
                for joint, point in item["skeleton"].items()
            }
        scaled.append(item)
    return scaled

class FullVideoPipeline:
    def __init__(
        self,
        target_side: str = "left",
        training_mode: str = "Free Bouting",
        model_checkpoint: str = "weights/fencenet/best_model.pth",
        max_pose_width: int | None = 960,
        pose_every_n_frames: int = 1,
    ):
        self.target_side = target_side
        self.training_mode = training_mode
        self.max_pose_width = max_pose_width
        self.pose_every_n_frames = max(1, int(pose_every_n_frames))
        self.pose_estimator = PoseEstimator(backend="ultralytics")
        self.target_tracker = TargetTracker(target_side=target_side)
        self.gatekeeper = ActivityGatekeeper(fps=30)
        self.sliding_window = SlidingWindowInference(model_path=model_checkpoint, device="auto")
        self.heuristics = HeuristicsEngine(target_side=target_side, training_mode=training_mode)
        self.normalizer = SpatialNormalizer()

    def _pose_frame(self, frame: np.ndarray) -> tuple[np.ndarray, float, float]:
        height, width = frame.shape[:2]
        if not self.max_pose_width or width <= self.max_pose_width:
            return frame, 1.0, 1.0

        scale = float(self.max_pose_width) / float(width)
        resized = cv2.resize(
            frame,
            (int(width * scale), int(height * scale)),
            interpolation=cv2.INTER_AREA,
        )
        return resized, float(width) / resized.shape[1], float(height) / resized.shape[0]
        
    def process_video(self, video_path: str, progress_callback=None) -> Dict[str, Any]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        effective_pose_every_n_frames = self.pose_every_n_frames
        if (
            total_frames
            and total_frames // effective_pose_every_n_frames < self.sliding_window.window_size
        ):
            effective_pose_every_n_frames = 1
        if progress_callback:
            progress_callback(0.05, "Reading video")
        
        frames_meta = []
        raw_skeletons = []
        normalized_skeletons = []
        active_frames_indices = []
        
        frame_idx = 0
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            should_extract = self.gatekeeper.should_extract_pose()
            should_extract = should_extract and (frame_idx % effective_pose_every_n_frames == 0)
            if should_extract:
                pose_frame, scale_x, scale_y = self._pose_frame(frame)
                detections = self.pose_estimator.extract_frame_fencers(pose_frame, persist_track=True)
                detections = scale_pose_detections(detections, scale_x, scale_y)
                target_skel, opp_skel = self.target_tracker.process_frame_detections(detections, frame_idx)
                
                is_active = self.gatekeeper.update(target_skel, opp_skel, width, self.target_side)
                
                frames_meta.append({
                    "frame_index": frame_idx,
                    "tracks": detections,
                    "target_skeleton": target_skel,
                    "target_bbox": self.target_tracker.last_known_bbox,
                    "target_track_id": self.target_tracker.locked_track_id,
                    "gatekeeper_state": self.gatekeeper.state,
                    "knee_angle": self.gatekeeper._get_knee_angle(target_skel, self.target_side) if target_skel else 180.0
                })
                
                if target_skel:
                    raw_skeletons.append(target_skel)
                    
                    # Convert pixel coords → xyn (0~1) to match training data format
                    xyn_skel = {k: (v[0] / frame_w, v[1] / frame_h) for k, v in target_skel.items()}
                    
                    if is_active:
                        try:
                            if self.normalizer.reference_nose is None:
                                self.normalizer.fit([xyn_skel])
                            norm_dict = self.normalizer.normalize_skeleton(xyn_skel)
                            norm_arr = np.array([norm_dict[j] for j in self.normalizer.MODEL_JOINT_NAMES])
                        except Exception as e:
                            logger.warning(f"Normalization failed: {e}")
                            norm_arr = np.zeros((9, 2))
                        normalized_skeletons.append(norm_arr)
                        active_frames_indices.append(frame_idx)
                    else:
                        normalized_skeletons.append(np.zeros((9, 2))) 
                        active_frames_indices.append(frame_idx)
                else:
                    raw_skeletons.append({})

            if progress_callback and total_frames and frame_idx % 30 == 0:
                fraction = min(0.82, 0.05 + 0.75 * (frame_idx / total_frames))
                progress_callback(fraction, f"Extracting pose ({frame_idx}/{total_frames})")
            
            frame_idx += 1
            
        cap.release()
        if progress_callback:
            progress_callback(0.84, "Classifying actions")
        
        if len(normalized_skeletons) > 0:
            skel_array = np.array(normalized_skeletons)
            action_segments_raw = self.sliding_window.run(skel_array)
            
            action_segments = []
            for seg in action_segments_raw:
                s_idx = min(seg["start_frame"], len(active_frames_indices)-1)
                e_idx = min(seg["end_frame"], len(active_frames_indices)-1)
                seg["video_start_frame"] = active_frames_indices[s_idx]
                seg["video_end_frame"] = active_frames_indices[e_idx]
                action_segments.append(seg)
        else:
            action_segments = []
            
        if progress_callback:
            progress_callback(0.88, "Checking posture")
        posture_errors = self.heuristics.evaluate(action_segments, raw_skeletons)
        
        for err in posture_errors:
            s_idx = min(err["start_frame"], len(active_frames_indices)-1)
            e_idx = min(err["end_frame"], len(active_frames_indices)-1)
            err["start_frame"] = active_frames_indices[s_idx]
            err["end_frame"] = active_frames_indices[e_idx]
            
        return {
            "training_mode": self.training_mode,
            "two_fencer_tracking": {
                "frames": frames_meta,
                "locked_track_id": self.target_tracker.locked_track_id
            },
            "action_segments": action_segments,
            "posture_errors": posture_errors
        }
