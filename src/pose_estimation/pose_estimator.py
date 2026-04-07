"""
Pose Estimator - Extract 2D skeleton from video frames.
Uses YOLO-Pose or similar model to detect skeletal keypoints.
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging

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
    
    def extract_frame_skeleton(
        self,
        frame: np.ndarray
    ) -> Optional[Dict[str, Tuple[float, float]]]:
        """
        Extract skeleton keypoints from a single frame.
        
        Args:
            frame: Input video frame (BGR format)
            
        Returns:
            Dictionary mapping joint names to (x, y) coordinates, or None if no
            usable person is detected
        """
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            raise ValueError("Frame must be a non-empty numpy array")

        if self.backend == "mock":
            return self._mock_skeleton(frame)

        if self.backend != "ultralytics" or self.model is None:
            raise RuntimeError(
                "Pose estimator is unavailable. Install ultralytics and provide a "
                "YOLO pose model, or use backend='mock' for tests."
            )

        results = self.model(frame, verbose=False)
        if not results:
            return None

        return self._extract_from_ultralytics_result(results[0])
    
    def extract_video_skeleton(self, video_path: str) -> List[Dict[str, Tuple[float, float]]]:
        """
        Extract skeleton keypoints for all frames in a video.
        
        Args:
            video_path: Path to input video file
            
        Returns:
            List of skeleton dictionaries, one per frame
        """
        skeletons = []
        
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
                
                skeleton = self.extract_frame_skeleton(frame)
                if skeleton is not None and self.validate_skeleton(skeleton):
                    skeletons.append(skeleton)
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
        
        return skeletons
    
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
        keypoints_obj = getattr(result, "keypoints", None)
        if keypoints_obj is None or getattr(keypoints_obj, "xy", None) is None:
            return None

        keypoints = self._to_numpy(keypoints_obj.xy)
        if keypoints.size == 0:
            return None
        if keypoints.ndim == 2:
            keypoints = np.expand_dims(keypoints, axis=0)

        confidences = None
        if getattr(keypoints_obj, "conf", None) is not None:
            confidences = self._to_numpy(keypoints_obj.conf)
            if confidences.ndim == 1:
                confidences = np.expand_dims(confidences, axis=0)

        person_idx = self._select_person_index(result, keypoints.shape[0])
        person_confidences = confidences[person_idx] if confidences is not None else None

        return self._build_skeleton_from_keypoints(
            keypoints=keypoints[person_idx],
            confidences=person_confidences
        )

    def _select_person_index(self, result: Any, num_people: int) -> int:
        """Select the largest detected person when boxes are available."""
        boxes = getattr(result, "boxes", None)
        xyxy = getattr(boxes, "xyxy", None) if boxes is not None else None
        if xyxy is None:
            return 0

        boxes_array = self._to_numpy(xyxy)
        if boxes_array.ndim != 2 or boxes_array.shape[0] != num_people:
            return 0

        widths = np.maximum(0.0, boxes_array[:, 2] - boxes_array[:, 0])
        heights = np.maximum(0.0, boxes_array[:, 3] - boxes_array[:, 1])
        areas = widths * heights
        return int(np.argmax(areas))

    def _build_skeleton_from_keypoints(
        self,
        keypoints: np.ndarray,
        confidences: Optional[np.ndarray] = None
    ) -> Optional[Dict[str, Tuple[float, float]]]:
        """Build the required skeleton dictionary from COCO keypoints."""
        skeleton = {}

        for joint_name, keypoint_idx in self.REQUIRED_JOINTS.items():
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

        return skeleton if self.validate_skeleton(skeleton) else None

    def _mock_skeleton(self, frame: np.ndarray) -> Dict[str, Tuple[float, float]]:
        """Generate a deterministic skeleton for tests and local pipeline smoke checks."""
        height, width = frame.shape[:2]
        center_x = width * 0.5
        head_y = height * 0.2
        shoulder_y = height * 0.35
        hip_y = height * 0.55
        knee_y = height * 0.7
        ankle_y = height * 0.85

        skeleton = {
            "nose": (center_x, head_y),
            "front_shoulder": (center_x + width * 0.06, shoulder_y),
            "front_elbow": (center_x + width * 0.13, height * 0.45),
            "front_wrist": (center_x + width * 0.20, height * 0.48),
            "left_hip": (center_x - width * 0.05, hip_y),
            "right_hip": (center_x + width * 0.05, hip_y),
            "left_knee": (center_x - width * 0.12, knee_y),
            "right_knee": (center_x + width * 0.12, knee_y),
            "left_ankle": (center_x - width * 0.18, ankle_y),
            "right_ankle": (center_x + width * 0.18, ankle_y),
        }
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
