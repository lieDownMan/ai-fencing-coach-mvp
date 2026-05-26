"""FenceNet sliding-window inference and full-video Python pipeline."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from inference.contracts import FrameSample, clean_skeleton
from src.data.fencing_dataset import CLASS_NAMES, MODEL_JOINTS, NUM_CHANNELS, SEQUENCE_LENGTH
from src.models.fencenet_v2 import FenceNetV2
from src.pose_estimation import PoseEstimator
from src.preprocessing import SpatialNormalizer

from .activity_gatekeeper import ActivityGatekeeper
from .heuristics_engine import HeuristicsEngine
from .target_tracker import TargetTracker

logger = logging.getLogger(__name__)

WINDOW_SIZE = SEQUENCE_LENGTH
STRIDE = 10
CONFIDENCE_THRESHOLD = 0.6


class SlidingWindowInference:
    """Classify normalized skeleton windows with FenceNetV2."""

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
        self.window_size = int(window_size)
        self.stride = int(stride)
        self.confidence_threshold = float(confidence_threshold)
        self.class_names = list(CLASS_NAMES)
        self.model = FenceNetV2(input_channels=NUM_CHANNELS)
        self.model_loaded = model_path is None
        self.model_error: Optional[str] = None
        self.model_path = str(model_path) if model_path is not None else None
        self.checkpoint_metadata: Dict[str, Any] = {}

        if model_path is not None:
            self._load_weights(model_path)
        self.model.to(self.device)
        self.model.eval()

    def run(self, skeleton_array: np.ndarray) -> List[Dict[str, Any]]:
        """Backward-compatible alias for classify_timeline()."""
        return self.classify_timeline(skeleton_array)

    def classify_window(
        self,
        window: np.ndarray,
        *,
        start_frame: int = 0,
        frame_indices: Optional[Sequence[int]] = None,
    ) -> Dict[str, Any]:
        """Classify one normalized ``(28, 9, 2)`` skeleton window."""
        self._ensure_model_available()
        window = self._validate_skeleton_array(
            window,
            expected_frames=self.window_size,
        )
        raw = self._classify_windows(window[np.newaxis, ...], starts=[start_frame])[0]
        if frame_indices is not None:
            raw.update(self._video_frame_range(frame_indices, start_frame, self.window_size))
        return raw

    def classify_timeline(
        self,
        skeleton_array: np.ndarray,
        *,
        frame_indices: Optional[Sequence[int]] = None,
    ) -> List[Dict[str, Any]]:
        """Classify a normalized timeline and merge overlapping predictions."""
        self._ensure_model_available()
        skeleton_array = self._validate_skeleton_array(skeleton_array)
        total_frames = skeleton_array.shape[0]
        if total_frames < self.window_size:
            logger.info(
                "Skipping FenceNet: %d active frames < window_size=%d.",
                total_frames,
                self.window_size,
            )
            return []

        starts = list(range(0, total_frames - self.window_size + 1, self.stride))
        raw_windows = self._classify_windows(skeleton_array, starts=starts)
        if frame_indices is not None:
            for window in raw_windows:
                window.update(
                    self._video_frame_range(
                        frame_indices,
                        window["start_frame"],
                        self.window_size,
                    )
                )
        return self._nms(raw_windows)

    def _classify_windows(
        self,
        skeleton_array: np.ndarray,
        starts: Optional[Sequence[int]] = None,
    ) -> List[Dict[str, Any]]:
        """Produce one raw prediction per sliding window."""
        skeleton_array = self._validate_skeleton_array(skeleton_array)
        if skeleton_array.ndim == 4:
            windows = skeleton_array
            if starts is None:
                starts = [0] * windows.shape[0]
        else:
            total_frames = skeleton_array.shape[0]
            if starts is None:
                starts = list(range(0, total_frames - self.window_size + 1, self.stride))
            windows = np.stack(
                [skeleton_array[start:start + self.window_size] for start in starts],
                axis=0,
            )

        if windows.shape[0] == 0:
            return []

        batch_size = windows.shape[0]
        flat = windows.reshape(batch_size, self.window_size, -1)
        tensor = (
            torch.from_numpy(flat)
            .float()
            .permute(0, 2, 1)
            .to(self.device)
        )

        with torch.no_grad():
            logits = self.model(tensor)
            probabilities = F.softmax(logits, dim=1)
            confidences, predictions = probabilities.max(dim=1)

        probs_np = probabilities.cpu().numpy()
        confs_np = confidences.cpu().numpy()
        preds_np = predictions.cpu().numpy()
        results: List[Dict[str, Any]] = []
        for index, start in enumerate(starts):
            class_idx = int(preds_np[index])
            confidence = float(confs_np[index])
            action = self.class_names[class_idx]
            reject_reason = None
            if confidence < self.confidence_threshold:
                reject_reason = "below_confidence_threshold"
                action = "Idle"
            results.append({
                "start_frame": int(start),
                "end_frame": int(start) + self.window_size,
                "action": action,
                "confidence": confidence,
                "class_idx": class_idx,
                "probabilities": {
                    self.class_names[i]: float(probs_np[index, i])
                    for i in range(len(self.class_names))
                },
                "reject_reason": reject_reason,
                "window_size": self.window_size,
                "stride": self.stride,
            })
        return results

    def _nms(self, windows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge consecutive overlapping windows with the same non-idle action."""
        if not windows:
            return []

        merged: List[Dict[str, Any]] = []
        current_group = [windows[0]]
        for window in windows[1:]:
            previous = current_group[-1]
            same_action = window["action"] == previous["action"]
            non_idle = window["action"] != "Idle"
            overlapping = int(window["start_frame"]) < int(previous["end_frame"])
            if same_action and non_idle and overlapping:
                current_group.append(window)
            else:
                merged.append(self._best_of_group(current_group))
                current_group = [window]
        merged.append(self._best_of_group(current_group))
        return [segment for segment in merged if segment["action"] != "Idle"]

    @staticmethod
    def _best_of_group(group: List[Dict[str, Any]]) -> Dict[str, Any]:
        best = max(group, key=lambda item: item["confidence"])
        segment = dict(best)
        segment["start_frame"] = int(group[0]["start_frame"])
        segment["end_frame"] = int(group[-1]["end_frame"])
        segment["merged_window_count"] = len(group)
        segment["merge_reason"] = (
            "same_action_overlap" if len(group) > 1 else "single_window"
        )
        if "video_start_frame" in group[0]:
            segment["video_start_frame"] = group[0]["video_start_frame"]
            segment["video_end_frame"] = group[-1]["video_end_frame"]
        return segment

    def _load_weights(self, model_path: str) -> None:
        path = Path(model_path)
        if not path.exists():
            self.model_loaded = False
            self.model_error = f"Model weights not found: {path}"
            logger.error(self.model_error)
            return

        try:
            checkpoint = torch.load(str(path), map_location=self.device, weights_only=True)
            state_dict, metadata = self._state_dict_and_metadata(checkpoint)
            self._validate_checkpoint_metadata(metadata)
            self.model.load_state_dict(state_dict)
        except Exception as exc:  # pragma: no cover - exact torch errors vary
            self.model_loaded = False
            self.model_error = f"Could not load FenceNet checkpoint {path}: {exc}"
            logger.error(self.model_error)
            return

        self.model_loaded = True
        self.model_error = None
        self.checkpoint_metadata = metadata
        logger.info("Loaded FenceNetV2 weights from %s", path)

    def _state_dict_and_metadata(self, checkpoint: Any) -> tuple[dict, Dict[str, Any]]:
        if not isinstance(checkpoint, dict):
            raise ValueError("checkpoint is not a dictionary")
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
            metadata = dict(checkpoint.get("metadata") or checkpoint)
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            metadata = dict(checkpoint.get("metadata") or checkpoint)
        elif self._looks_like_state_dict(checkpoint):
            state_dict = checkpoint
            metadata = {}
        else:
            raise ValueError("unrecognized checkpoint format")
        return state_dict, metadata

    def _validate_checkpoint_metadata(self, metadata: Dict[str, Any]) -> None:
        if not metadata:
            return
        input_channels = metadata.get("input_channels")
        if input_channels is not None and int(input_channels) != NUM_CHANNELS:
            raise ValueError(
                f"input_channels mismatch: {input_channels} != {NUM_CHANNELS}"
            )
        num_classes = metadata.get("num_classes")
        if num_classes is not None and int(num_classes) != len(self.class_names):
            raise ValueError(
                f"num_classes mismatch: {num_classes} != {len(self.class_names)}"
            )
        action_classes = metadata.get("action_classes")
        if action_classes is not None and list(action_classes) != self.class_names:
            raise ValueError("action_classes mismatch")
        sequence_length = metadata.get("sequence_length") or metadata.get("window_size")
        if sequence_length is not None and int(sequence_length) != self.window_size:
            raise ValueError(
                f"sequence length mismatch: {sequence_length} != {self.window_size}"
            )

    def _ensure_model_available(self) -> None:
        if not self.model_loaded:
            raise RuntimeError(self.model_error or "FenceNet model is unavailable")

    @staticmethod
    def _looks_like_state_dict(item: dict) -> bool:
        return bool(item) and all(hasattr(value, "shape") for value in item.values())

    def _validate_skeleton_array(
        self,
        skeleton_array: np.ndarray,
        *,
        expected_frames: Optional[int] = None,
    ) -> np.ndarray:
        array = np.asarray(skeleton_array, dtype=np.float32)
        if array.ndim == 4:
            if array.shape[1:] != (self.window_size, len(MODEL_JOINTS), 2):
                raise ValueError(
                    "Expected batched windows with shape "
                    f"(N, {self.window_size}, {len(MODEL_JOINTS)}, 2)"
                )
        elif array.ndim == 3:
            if array.shape[1:] != (len(MODEL_JOINTS), 2):
                raise ValueError(
                    f"Expected skeleton array with shape (T, {len(MODEL_JOINTS)}, 2)"
                )
            if expected_frames is not None and array.shape[0] != expected_frames:
                raise ValueError(
                    f"Expected {expected_frames} frames, got {array.shape[0]}"
                )
        else:
            raise ValueError("Skeleton array must be rank 3 or rank 4")
        if not np.all(np.isfinite(array)):
            raise ValueError("Skeleton array contains non-finite values")
        return array

    @staticmethod
    def _video_frame_range(
        frame_indices: Sequence[int],
        start_frame: int,
        window_size: int,
    ) -> Dict[str, int]:
        start_index = min(int(start_frame), len(frame_indices) - 1)
        end_index = min(int(start_frame) + int(window_size) - 1, len(frame_indices) - 1)
        return {
            "video_start_frame": int(frame_indices[start_index]),
            "video_end_frame": int(frame_indices[end_index]),
        }


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
        for skeleton_key in ("skeleton", "raw_skeleton"):
            if skeleton_key in item and item[skeleton_key] is not None:
                item[skeleton_key] = {
                    joint: (float(point[0]) * scale_x, float(point[1]) * scale_y)
                    for joint, point in item[skeleton_key].items()
                }
        scaled.append(item)
    return scaled


class FullVideoPipeline:
    """End-to-end clip pipeline for Python local analysis."""

    def __init__(
        self,
        target_side: str = "left",
        training_mode: str = "Free Bouting",
        model_checkpoint: str = "weights/fencenet/best_model.pth",
        max_pose_width: int | None = 960,
        pose_every_n_frames: int = 1,
        handedness: str = "auto",
    ):
        self.target_side = target_side
        self.training_mode = training_mode
        self.max_pose_width = max_pose_width
        self.pose_every_n_frames = max(1, int(pose_every_n_frames))
        self.pose_estimator = PoseEstimator(
            backend="ultralytics",
            target_side=target_side,
            handedness=handedness,
        )
        self.target_tracker = TargetTracker(target_side=target_side)
        self.gatekeeper = ActivityGatekeeper(fps=30)
        self.sliding_window = SlidingWindowInference(
            model_path=model_checkpoint,
            device="auto",
        )
        self.heuristics = HeuristicsEngine(
            target_side=target_side,
            training_mode=training_mode,
        )

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

    def _pixel_to_xyn(
        self,
        skeleton: Dict[str, Any],
        frame_h: int,
        frame_w: int,
    ) -> Dict[str, tuple[float, float]]:
        return {
            key: (float(value[0]) / frame_w, float(value[1]) / frame_h)
            for key, value in skeleton.items()
            if isinstance(value, (list, tuple, np.ndarray)) and len(value) == 2
        }

    def _normalize_active_skeleton(
        self,
        skeleton: Dict[str, Any],
        normalizer: SpatialNormalizer,
        frame_h: int,
        frame_w: int,
    ) -> Optional[np.ndarray]:
        xyn_skeleton = self._pixel_to_xyn(skeleton, frame_h, frame_w)
        if normalizer.reference_nose is None:
            normalizer.fit([xyn_skeleton])
        normalized = normalizer.normalize_skeleton(xyn_skeleton)
        return np.array([normalized[joint] for joint in MODEL_JOINTS], dtype=np.float32)

    def process_video(self, video_path: str, progress_callback=None) -> Dict[str, Any]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        frame_w = width
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.gatekeeper = ActivityGatekeeper(fps=int(round(fps)) or 30)

        effective_pose_every_n_frames = self.pose_every_n_frames
        if total_frames and total_frames // effective_pose_every_n_frames < self.sliding_window.window_size:
            effective_pose_every_n_frames = 1
        if progress_callback:
            progress_callback(0.05, "Reading video")

        frame_samples: List[FrameSample] = []
        active_runs: List[List[FrameSample]] = []
        current_run: List[FrameSample] = []
        normalizer = SpatialNormalizer()
        was_active = False
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            should_extract = self.gatekeeper.should_extract_pose()
            should_extract = should_extract and (frame_idx % effective_pose_every_n_frames == 0)
            if should_extract:
                pose_frame, scale_x, scale_y = self._pose_frame(frame)
                detections = self.pose_estimator.extract_frame_fencers(
                    pose_frame,
                    persist_track=True,
                )
                detections = scale_pose_detections(detections, scale_x, scale_y)
                target_skel, opponent_skel = self.target_tracker.process_frame_detections(
                    detections,
                    frame_idx,
                )
                is_active = self.gatekeeper.update(
                    target_skel,
                    opponent_skel,
                    width,
                    self.target_side,
                )

                sample = FrameSample(
                    frame_index=frame_idx,
                    timestamp=float(frame_idx / fps),
                    tracks=detections,
                    target_detection=self.target_tracker.last_target_detection,
                    opponent_detection=self.target_tracker.last_opponent_detection,
                    target_skeleton=clean_skeleton(target_skel),
                    opponent_skeleton=clean_skeleton(opponent_skel),
                    gate_state=self.gatekeeper.state,
                    gate_reasons=dict(self.gatekeeper.last_reasons),
                    interpolated=bool(self.target_tracker.last_interpolated),
                )

                if is_active and target_skel:
                    if not was_active:
                        normalizer = SpatialNormalizer()
                        current_run = []
                    try:
                        sample.normalized_skeleton = self._normalize_active_skeleton(
                            target_skel,
                            normalizer,
                            frame_h,
                            frame_w,
                        )
                        sample.valid_for_model = True
                        sample.active_index = len(current_run)
                        current_run.append(sample)
                    except Exception as exc:
                        logger.warning("Normalization failed on frame %s: %s", frame_idx, exc)
                elif was_active and current_run:
                    active_runs.append(current_run)
                    current_run = []

                frame_samples.append(sample)
                was_active = bool(is_active)

            if progress_callback and total_frames and frame_idx % 30 == 0:
                fraction = min(0.82, 0.05 + 0.75 * (frame_idx / total_frames))
                progress_callback(fraction, f"Extracting pose ({frame_idx}/{total_frames})")
            frame_idx += 1

        cap.release()
        if current_run:
            active_runs.append(current_run)

        if progress_callback:
            progress_callback(0.84, "Classifying actions")

        action_segments: List[Dict[str, Any]] = []
        posture_errors: List[Dict[str, Any]] = []
        for run_index, run in enumerate(active_runs):
            valid_samples = [
                sample for sample in run
                if sample.valid_for_model and sample.normalized_skeleton is not None
            ]
            if len(valid_samples) < self.sliding_window.window_size:
                continue
            skeleton_array = np.stack(
                [sample.normalized_skeleton for sample in valid_samples],
                axis=0,
            )
            frame_indices = [sample.frame_index for sample in valid_samples]
            segments = self.sliding_window.classify_timeline(
                skeleton_array,
                frame_indices=frame_indices,
            )
            raw_skeletons = [sample.target_skeleton for sample in valid_samples]
            for segment in segments:
                segment["active_run_index"] = run_index
                action_segments.append(segment)

            run_errors = self.heuristics.evaluate(segments, raw_skeletons)
            for error in run_errors:
                start_index = min(int(error["start_frame"]), len(frame_indices) - 1)
                end_index = min(max(int(error["end_frame"]) - 1, start_index), len(frame_indices) - 1)
                error["active_run_index"] = run_index
                error["active_start_frame"] = int(error["start_frame"])
                error["active_end_frame"] = int(error["end_frame"])
                error["start_frame"] = int(frame_indices[start_index])
                error["end_frame"] = int(frame_indices[end_index])
                posture_errors.append(error)

        if progress_callback:
            progress_callback(0.88, "Checking posture")

        return {
            "training_mode": self.training_mode,
            "two_fencer_tracking": {
                "frames": [sample.to_report_dict() for sample in frame_samples],
                "locked_track_id": self.target_tracker.locked_track_id,
                "lock_state": self.target_tracker.lock_state,
            },
            "action_segments": action_segments,
            "posture_errors": posture_errors,
        }
