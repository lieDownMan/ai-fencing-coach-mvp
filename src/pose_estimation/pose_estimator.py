"""
Pose Estimator - Extract 2D skeleton from video frames.
Uses YOLO-Pose or similar model to detect skeletal keypoints.
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging

from ..fencing_skeleton import canonicalize_front_joints
from ..tracking import FencerTracker

logger = logging.getLogger(__name__)


class PoseEstimator:
    """
    Extracts 2D pose keypoints from video frames.
    Required joints: front wrist, front elbow, front shoulder, 
                    both hips, both knees, both ankles, nose, front ankle.
    """
    
    # Keypoint indices for YOLO-Pose (17 joints)
    KEYPOINT_NAMES = {
        0: "nose",
        1: "left_eye", 2: "right_eye",
        3: "left_ear", 4: "right_ear",
        5: "left_shoulder", 6: "right_shoulder",
        7: "left_elbow", 8: "right_elbow",
        9: "left_wrist", 10: "right_wrist",
        11: "left_hip", 12: "right_hip",
        13: "left_knee", 14: "right_knee",
        15: "left_ankle", 16: "right_ankle",
    }

    ANATOMICAL_JOINTS = {
        "nose": 0,
        "left_shoulder": 5,
        "right_shoulder": 6,
        "left_elbow": 7,
        "right_elbow": 8,
        "left_wrist": 9,
        "right_wrist": 10,
        "left_hip": 11,
        "right_hip": 12,
        "left_knee": 13,
        "right_knee": 14,
        "left_ankle": 15,
        "right_ankle": 16,
    }
    
    # Mapping for fencing-specific joints (right-handed fencer assumed)
    REQUIRED_JOINTS = {
        "nose": 0,
        "front_wrist": 10,  # right wrist
        "front_elbow": 8,   # right elbow
        "front_shoulder": 6,  # right shoulder
        "left_hip": 11,
        "right_hip": 12,
        "left_knee": 13,
        "right_knee": 14,
        "left_ankle": 15,
        "right_ankle": 16,
        "front_ankle": 16,  # right ankle
    }

    DEFAULT_MODEL_PATH = "yolov8n-pose.pt"
    SUPPORTED_BACKENDS = {"auto", "ultralytics", "mock"}
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        conf_threshold: float = 0.5,
        backend: str = "auto"
    ):
        """
        Initialize Pose Estimator.
        
        Args:
            model_path: Path to YOLO or pose model weights
            conf_threshold: Confidence threshold for keypoint detection
            backend: "auto", "ultralytics", or "mock"
        """
        if backend not in self.SUPPORTED_BACKENDS:
            raise ValueError(
                f"Unsupported pose backend: {backend}. "
                f"Supported: {sorted(self.SUPPORTED_BACKENDS)}"
            )

        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.requested_backend = backend
        self.backend = backend
        self.model = self._load_model()
        self.fencer_tracker = FencerTracker()
        self._tracking_fallback_warned = False
        self._tracking_runtime_available = True
        
    def _load_model(self):
        """Load pose estimation model if a real backend is available."""
        if self.requested_backend == "mock":
            logger.info("Using mock pose estimator backend.")
            return None

        try:
            from ultralytics import YOLO
        except ModuleNotFoundError as exc:
            if self.requested_backend == "ultralytics":
                raise RuntimeError(
                    "Ultralytics is not installed. Install it or use backend='mock' "
                    "for tests."
                ) from exc
            self.backend = "unavailable"
            logger.warning("Ultralytics is not installed; pose extraction is unavailable.")
            return None

        model_path = self.model_path or self.DEFAULT_MODEL_PATH
        try:
            model = YOLO(model_path)
            self.backend = "ultralytics"
            logger.info(f"Pose estimation model loaded: {model_path}")
            return model
        except Exception as exc:
            if self.requested_backend == "ultralytics":
                raise RuntimeError(f"Failed to load pose model: {model_path}") from exc
            self.backend = "unavailable"
            logger.warning(f"Could not load pose model {model_path}: {exc}")
            return None

    def is_available(self) -> bool:
        """Return whether this estimator can produce skeletons."""
        return self.backend in {"ultralytics", "mock"}
    
    def _validate_frame(self, frame: np.ndarray):
        """Validate a frame before pose extraction."""
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            raise ValueError("Frame must be a non-empty numpy array")

    def extract_frame_fencers(
        self,
        frame: np.ndarray,
        persist_track: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Extract up to all valid fencer candidates from a single frame.

        Returns detections sorted by size, largest first. A downstream tracker
        can then select the two largest and assign left/right labels.
        """
        self._validate_frame(frame)

        if self.backend == "mock":
            return self._mock_fencer_detections(frame)

        if self.backend != "ultralytics" or self.model is None:
            raise RuntimeError(
                "Pose estimator is unavailable. Install ultralytics and provide a "
                "YOLO pose model, or use backend='mock' for tests."
            )

        if persist_track and self._tracking_runtime_available:
            try:
                results = self.model.track(
                    frame,
                    verbose=False,
                    persist=True,
                    tracker="bytetrack.yaml",
                )
            except ModuleNotFoundError as exc:
                if "lap" not in str(exc):
                    raise
                self._tracking_runtime_available = False
                if not self._tracking_fallback_warned:
                    logger.warning(
                        "Ultralytics tracking dependency 'lap' is missing; "
                        "falling back to per-frame pose detection without ByteTrack IDs."
                    )
                    self._tracking_fallback_warned = True
                results = self.model(frame, verbose=False)
            except Exception as exc:
                self._tracking_runtime_available = False
                if not self._tracking_fallback_warned:
                    logger.warning(
                        "Ultralytics track() failed (%s); falling back to per-frame pose detection.",
                        exc,
                    )
                    self._tracking_fallback_warned = True
                results = self.model(frame, verbose=False)
        else:
            results = self.model(frame, verbose=False)
            
        if not results:
            return []

        detections = self._extract_fencer_detections_from_ultralytics_result(results[0])
        if persist_track and detections and not any(
            detection.get("track_id") is not None for detection in detections
        ):
            detections = self._assign_fallback_track_ids(detections)
        return detections

    def extract_frame_skeleton(
        self,
        frame: np.ndarray
    ) -> Optional[Dict[str, Tuple[float, float]]]:
        """
        Extract one selected skeleton from a single frame.

        The selected skeleton is the largest valid pose candidate, preserving the
        original single-fencer inference path while two-fencer tracking records
        additional candidates separately.
        """
        detections = self.extract_frame_fencers(frame)
        if not detections:
            return None
        selected = detections[0]
        center_x = float((selected.get("center") or [frame.shape[1] * 0.5])[0])
        screen_side = "left" if center_x < (frame.shape[1] * 0.5) else "right"
        return canonicalize_front_joints(selected["skeleton"], screen_side=screen_side)

    def extract_video_fencer_tracks(self, video_path: str) -> Dict[str, Any]:
        """
        Extract primary skeletons and side-based two-fencer tracks for a video.

        Returns a payload with `skeletons` for the existing classifier path and
        `frames`/`summary` for visualization and reporting.
        """
        skeletons = []
        tracking_frames = []

        cap = None
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Cannot open video: {video_path}")

            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                detections = self.extract_frame_fencers(frame)
                if detections:
                    skeleton = canonicalize_front_joints(
                        detections[0]["skeleton"],
                        screen_side=(
                            "left"
                            if float((detections[0].get("center") or [frame.shape[1] * 0.5])[0])
                            < (frame.shape[1] * 0.5)
                            else "right"
                        ),
                    )
                    if self.validate_skeleton(skeleton):
                        skeletons.append(skeleton)
                tracking_frames.append(
                    self.fencer_tracker.build_frame(frame_count, detections)
                )
                frame_count += 1

                if frame_count % 100 == 0:
                    logger.info(f"Processed {frame_count} frames from {video_path}")

            logger.info(f"Video processing complete: {frame_count} frames extracted")

        except Exception as e:
            logger.error(f"Error extracting skeleton from video: {e}")
            raise
        finally:
            if cap is not None:
                cap.release()

        payload = self.fencer_tracker.build_payload(tracking_frames)
        payload["skeletons"] = skeletons
        return payload

    def extract_video_skeleton(self, video_path: str) -> List[Dict[str, Tuple[float, float]]]:
        """
        Extract skeleton keypoints for all frames in a video.

        Args:
            video_path: Path to input video file

        Returns:
            List of skeleton dictionaries, one per valid frame
        """
        return self.extract_video_fencer_tracks(video_path)["skeletons"]

    def validate_skeleton(self, skeleton: Dict[str, Tuple[float, float]]) -> bool:
        """
        Validate that all required joints are present in skeleton.
        
        Args:
            skeleton: Skeleton dictionary
            
        Returns:
            True if all required joints are present
        """
        if not all(joint in skeleton for joint in self.REQUIRED_JOINTS.keys()):
            return False

        for coords in skeleton.values():
            try:
                if len(coords) != 2:
                    return False
            except TypeError:
                return False
            if not np.isfinite(coords[0]) or not np.isfinite(coords[1]):
                return False

        return True

    def _extract_from_ultralytics_result(
        self,
        result: Any
    ) -> Optional[Dict[str, Tuple[float, float]]]:
        """Extract one fencing skeleton from an Ultralytics result."""
        detections = self._extract_fencer_detections_from_ultralytics_result(result)
        if not detections:
            return None
        return detections[0]["skeleton"]

    def _extract_fencer_detections_from_ultralytics_result(
        self,
        result: Any
    ) -> List[Dict[str, Any]]:
        """Extract valid fencer candidates from an Ultralytics result."""
        keypoints_obj = getattr(result, "keypoints", None)
        if keypoints_obj is None or getattr(keypoints_obj, "xy", None) is None:
            return []

        keypoints = self._to_numpy(keypoints_obj.xy)
        if keypoints.size == 0:
            return []
        if keypoints.ndim == 2:
            keypoints = np.expand_dims(keypoints, axis=0)

        confidences = None
        if getattr(keypoints_obj, "conf", None) is not None:
            confidences = self._to_numpy(keypoints_obj.conf)
            if confidences.ndim == 1:
                confidences = np.expand_dims(confidences, axis=0)

        boxes = self._extract_boxes(result, keypoints.shape[0])
        box_confidences = self._extract_box_confidences(result, keypoints.shape[0])
        
        track_ids = None
        result_boxes = getattr(result, "boxes", None)
        if result_boxes is not None and getattr(result_boxes, "id", None) is not None:
            track_ids = self._to_numpy(result_boxes.id)
            
        detections = []

        for person_idx in range(keypoints.shape[0]):
            person_confidences = (
                confidences[person_idx] if confidences is not None else None
            )
            skeleton = self._build_skeleton_from_keypoints(
                keypoints=keypoints[person_idx],
                confidences=person_confidences
            )
            if skeleton is None:
                continue

            confidence = self._detection_confidence(
                person_confidences,
                box_confidences[person_idx] if box_confidences is not None else None
            )
            candidate = self.fencer_tracker.candidate_from_skeleton(
                skeleton=skeleton,
                confidence=confidence,
                source_rank=person_idx
            )
            if boxes is not None:
                bbox = [float(value) for value in boxes[person_idx][:4]]
                candidate["bbox"] = bbox
                candidate["center"] = [
                    float((bbox[0] + bbox[2]) / 2.0),
                    float((bbox[1] + bbox[3]) / 2.0),
                ]
                candidate["area"] = self._bbox_area(bbox)
                
            if track_ids is not None and person_idx < len(track_ids):
                candidate["track_id"] = int(track_ids[person_idx])
                
            detections.append(candidate)

        detections.sort(key=lambda detection: detection.get("area", 0.0), reverse=True)
        for source_rank, detection in enumerate(detections):
            detection["source_rank"] = source_rank
        return detections

    def _extract_boxes(
        self,
        result: Any,
        num_people: int
    ) -> Optional[np.ndarray]:
        """Return Ultralytics boxes as an (N, 4) array when available."""
        boxes = getattr(result, "boxes", None)
        xyxy = getattr(boxes, "xyxy", None) if boxes is not None else None
        if xyxy is None:
            return None

        boxes_array = self._to_numpy(xyxy)
        if (
            boxes_array.ndim != 2
            or boxes_array.shape[0] != num_people
            or boxes_array.shape[1] < 4
        ):
            return None
        return boxes_array[:, :4]

    def _extract_box_confidences(
        self,
        result: Any,
        num_people: int
    ) -> Optional[np.ndarray]:
        """Return Ultralytics detection confidences when available."""
        boxes = getattr(result, "boxes", None)
        confidence_values = getattr(boxes, "conf", None) if boxes is not None else None
        if confidence_values is None:
            return None

        confidences = self._to_numpy(confidence_values)
        if confidences.ndim != 1 or confidences.shape[0] != num_people:
            return None
        return confidences

    def _detection_confidence(
        self,
        keypoint_confidences: Optional[np.ndarray],
        box_confidence: Optional[float]
    ) -> float:
        """Return one confidence score for a fencer candidate."""
        if box_confidence is not None and np.isfinite(box_confidence):
            return float(box_confidence)
        if keypoint_confidences is None:
            return 1.0

        required_indices = [
            keypoint_idx
            for keypoint_idx in set(self.REQUIRED_JOINTS.values()) | set(self.ANATOMICAL_JOINTS.values())
            if keypoint_idx < keypoint_confidences.shape[0]
        ]
        values = keypoint_confidences[required_indices]
        finite_values = values[np.isfinite(values)]
        if finite_values.size == 0:
            return 0.0
        return float(np.mean(finite_values))

    @staticmethod
    def _bbox_area(bbox: List[float]) -> float:
        """Return bbox area in pixels."""
        width = max(0.0, float(bbox[2]) - float(bbox[0]))
        height = max(0.0, float(bbox[3]) - float(bbox[1]))
        return float(width * height)

    def _build_skeleton_from_keypoints(
        self,
        keypoints: np.ndarray,
        confidences: Optional[np.ndarray] = None
    ) -> Optional[Dict[str, Tuple[float, float]]]:
        """Build the required skeleton dictionary from COCO keypoints."""
        skeleton = {}

        for joint_name, keypoint_idx in self.ANATOMICAL_JOINTS.items():
            if keypoint_idx >= keypoints.shape[0]:
                return None
            if (
                confidences is not None
                and keypoint_idx < confidences.shape[0]
                and confidences[keypoint_idx] < self.conf_threshold
            ):
                return None

            x_coord, y_coord = keypoints[keypoint_idx][:2]
            if not np.isfinite(x_coord) or not np.isfinite(y_coord):
                return None

            skeleton[joint_name] = (float(x_coord), float(y_coord))

        skeleton["front_shoulder"] = skeleton["right_shoulder"]
        skeleton["front_elbow"] = skeleton["right_elbow"]
        skeleton["front_wrist"] = skeleton["right_wrist"]
        skeleton["front_ankle"] = skeleton["right_ankle"]

        return skeleton if self.validate_skeleton(skeleton) else None

    def _mock_fencer_detections(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Generate two deterministic fencer candidates for tests and demos."""
        height, width = frame.shape[:2]
        left_skeleton = self._mock_skeleton(frame, center_x=width * 0.36)
        right_skeleton = self._mock_skeleton(frame, center_x=width * 0.64)
        left_candidate = self.fencer_tracker.candidate_from_skeleton(
            left_skeleton,
            confidence=1.0,
            source_rank=0
        )
        right_candidate = self.fencer_tracker.candidate_from_skeleton(
            right_skeleton,
            confidence=1.0,
            source_rank=1
        )
        left_candidate["track_id"] = 1
        right_candidate["track_id"] = 2
        return [left_candidate, right_candidate]

    @staticmethod
    def _assign_fallback_track_ids(
        detections: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Assign deterministic left-to-right track IDs when ByteTrack is unavailable.

        This is not true identity tracking, but it keeps the downstream target
        tracker and visualizer usable in restricted environments.
        """
        ordered = sorted(
            detections,
            key=lambda detection: float((detection.get("center") or [0.0])[0]),
        )
        for index, detection in enumerate(ordered, start=1):
            detection["track_id"] = index
        return ordered

    def _mock_skeleton(
        self,
        frame: np.ndarray,
        center_x: Optional[float] = None
    ) -> Dict[str, Tuple[float, float]]:
        """Generate a deterministic skeleton for tests and local pipeline smoke checks."""
        height, width = frame.shape[:2]
        if center_x is None:
            center_x = width * 0.5
        head_y = height * 0.2
        shoulder_y = height * 0.35
        hip_y = height * 0.55
        knee_y = height * 0.7
        ankle_y = height * 0.85

        skeleton = {
            "nose": (center_x, head_y),
            "left_shoulder": (center_x - width * 0.06, shoulder_y),
            "right_shoulder": (center_x + width * 0.06, shoulder_y),
            "left_elbow": (center_x - width * 0.13, height * 0.45),
            "right_elbow": (center_x + width * 0.13, height * 0.45),
            "left_wrist": (center_x - width * 0.20, height * 0.48),
            "right_wrist": (center_x + width * 0.20, height * 0.48),
            "left_hip": (center_x - width * 0.05, hip_y),
            "right_hip": (center_x + width * 0.05, hip_y),
            "left_knee": (center_x - width * 0.12, knee_y),
            "right_knee": (center_x + width * 0.12, knee_y),
            "left_ankle": (center_x - width * 0.18, ankle_y),
            "right_ankle": (center_x + width * 0.18, ankle_y),
        }
        skeleton["front_shoulder"] = skeleton["right_shoulder"]
        skeleton["front_elbow"] = skeleton["right_elbow"]
        skeleton["front_wrist"] = skeleton["right_wrist"]
        skeleton["front_ankle"] = skeleton["right_ankle"]
        return skeleton

    @staticmethod
    def _to_numpy(value: Any) -> np.ndarray:
        """Convert tensors or array-like values to numpy arrays."""
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            return value.numpy()
        return np.asarray(value)
