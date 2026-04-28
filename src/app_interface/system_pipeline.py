"""
System Pipeline - Orchestrates the complete fencing coaching system.
Coordinates: Pose Estimation -> Preprocessing -> FenceNetV2 -> Heuristics -> Tracking -> LLM Coaching
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import logging

from ..pose_estimation import PoseEstimator
from ..preprocessing import SpatialNormalizer, TemporalSampler
from ..models import FenceNet, FenceNetV2, BiFenceNet
from ..inference.sliding_window import SlidingWindowInference
from ..inference.heuristics_engine import HeuristicsEngine
from ..tracking import FencerTracker, PatternAnalyzer, ProfileManager
from ..llm_agent import CoachEngine
from ..fencing_skeleton import normalize_weapon_hand

logger = logging.getLogger(__name__)


class SystemPipeline:
    """
    Main orchestrator for the fencing coaching system.
    Manages all pipeline stages from video input to coaching output.
    """

    SUPPORTED_CHECKPOINT_FORMAT_VERSION = 1
    INFERENCE_WINDOW_SIZE = 28
    INFERENCE_STRIDE = 10
    STEP_ACTIONS = {"SF", "SB"}
    STEP_TRACKING_MOTION_MIN_PX = 20.0
    
    def __init__(
        self,
        use_bifencenet: bool = False,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        model_checkpoint: Optional[str] = None,
        profiles_dir: str = "data/fencer_profiles/",
        pose_backend: str = "auto",
        pose_model_path: Optional[str] = None,
        llm_model_name: str = "llava-next",
        target_side: str = "left",
        weapon_hand: str = "auto",
    ):
        """
        Initialize System Pipeline.
        
        Args:
            use_bifencenet: Use BiFenceNet instead of FenceNet
            device: Device to use for models (cuda or cpu)
            model_checkpoint: Path to pretrained model weights
            profiles_dir: Directory for fencer profiles
            pose_backend: Pose estimator backend ("auto", "ultralytics", or "mock")
            pose_model_path: Optional pose model path
            llm_model_name: CoachEngine LLM model name
            target_side: Which fencer to analyze based on screen side
            weapon_hand: Weapon-hand override for the target fencer ("auto", "left", or "right")
        """
        self.device = device
        self.use_bifencenet = use_bifencenet
        self.model_checkpoint_path = (
            str(Path(model_checkpoint).expanduser())
            if model_checkpoint
            else None
        )
        self.model_checkpoint_loaded = False
        self.model_checkpoint_metadata: Dict[str, Any] = {}
        self.model_checkpoint_error: Optional[str] = None
        
        logger.info(f"Initializing System Pipeline on device: {device}")
        
        # Phase 1: Pose Estimation
        self.pose_estimator = PoseEstimator(
            model_path=pose_model_path,
            backend=pose_backend
        )
        logger.info(f"Using {pose_backend}, {pose_model_path} pose estimation")
        
        # Phase 2: Preprocessing
        self.spatial_normalizer = SpatialNormalizer()
        self.temporal_sampler = TemporalSampler(
            target_length=self.INFERENCE_WINDOW_SIZE
        )
        self.model_joint_names = list(SpatialNormalizer.MODEL_JOINT_NAMES)
        self.model_input_channels = len(self.model_joint_names) * 2
        
        # Phase 3: FenceNet Model
        if use_bifencenet:
            self.model = BiFenceNet(
                input_channels=self.model_input_channels,
                device=device
            ).to(device)
            logger.info("Using BiFenceNet model")
        else:
            self.model = FenceNetV2(
                input_channels=self.model_input_channels,
            ).to(device)
            logger.info("Using FenceNetV2 model")
        
        self._load_model_checkpoint()
        
        self.model.eval()

        self.target_side = target_side
        self.weapon_hand = normalize_weapon_hand(weapon_hand)

        # Phase 3b: Sliding Window Inference (wraps its own FenceNetV2 copy)
        self.sliding_window = SlidingWindowInference(
            model_path=self.model_checkpoint_path,
            device=device,
            window_size=self.INFERENCE_WINDOW_SIZE,
            stride=self.INFERENCE_STRIDE,
            target_side=self.target_side,
        )

        # Phase 3c: Geometric Heuristics Engine
        self.heuristics_engine = HeuristicsEngine(target_side=self.target_side)
        
        # Phase 4: Pattern Tracking
        self.fencer_tracker = FencerTracker(
            target_side=self.target_side,
            weapon_hand=self.weapon_hand,
        )
        self.pattern_analyzer = PatternAnalyzer()
        self.profile_manager = ProfileManager(profiles_dir=profiles_dir)
        
        # Phase 5: LLM Coaching
        self.coach_engine = CoachEngine(
            model_name=llm_model_name,
            profiles_dir=profiles_dir,
            device=device
        )
        
        # Runtime state
        self.current_bout_stats = {}
        self.current_fencer_id = None
        self.current_opponent_id = None
        self.current_score = {"player": 0, "opponent": 0}
    
    def process_video(
        self,
        video_path: str,
        fencer_id: str,
        batch_process: bool = True
    ) -> Dict[str, Any]:
        """
        Process complete video through entire pipeline.
        
        Args:
            video_path: Path to input video
            fencer_id: Fencer identifier
            batch_process: Whether to process frames in batches
            
        Returns:
            Processing results dictionary
        """
        logger.info(f"Processing video: {video_path}")
        
        self.current_fencer_id = fencer_id
        self.pattern_analyzer.clear_history()
        results = {
            "video_path": video_path,
            "fencer_id": fencer_id,
            "frames_processed": 0,
            "classifications": [],
            "action_segments": [],
            "posture_errors": [],
            "statistics": {},
            "window_size": self.INFERENCE_WINDOW_SIZE,
            "window_stride": self.INFERENCE_STRIDE,
            "two_fencer_tracking": self.fencer_tracker.build_payload([]),
        }
        
        try:
            # Phase 1: Extract Skeleton
            logger.info("Phase 1: Extracting skeleton...")
            skeletons, tracking_payload = self._extract_pose_sequence(video_path)
            results["two_fencer_tracking"] = tracking_payload
            results["frames_processed"] = len(skeletons)
            
            if not skeletons:
                logger.error("No skeletons extracted from video")
                return results
            
            # Phase 2: Preprocess Skeleton
            logger.info("Phase 2: Preprocessing skeleton...")
            self.spatial_normalizer.fit(skeletons)
            normalized_skeletons = self.spatial_normalizer.normalize_sequence(skeletons)
            skeleton_array = self.spatial_normalizer.get_normalized_array(
                normalized_skeletons,
                joint_names=self.model_joint_names,
                already_normalized=True
            )
            if skeleton_array.shape[0] < self.temporal_sampler.target_length:
                skeleton_array = self.temporal_sampler.sample_array(skeleton_array)
            
            # Phase 3a: Legacy Model Inference (kept for backward compat)
            logger.info("Phase 3a: Running FenceNet inference (legacy)...")
            classifications = self._run_inference(skeleton_array, batch_process)
            results["classifications"] = classifications
            
            # Phase 3b: Sliding Window Action Spotting (new)
            logger.info("Phase 3b: Running sliding window action spotting...")
            action_segments = self.sliding_window.run(skeleton_array)
            
            # Map action segments bounds back to actual video frames
            frame_map = tracking_payload.get("active_to_video_map", [])
            for seg in action_segments:
                start_idx = seg["start_frame"]
                end_idx = min(seg["end_frame"] - 1, len(frame_map) - 1)
                if start_idx < len(frame_map) and end_idx >= 0:
                    seg["video_start_frame"] = frame_map[start_idx]
                    seg["video_end_frame"  ] = frame_map[end_idx]

            action_segments = self._refine_step_segments_with_tracking(
                action_segments,
                tracking_payload,
            )
            
            results["action_segments"] = action_segments
            logger.info("Detected %d action segments", len(action_segments))
            
            # Phase 3c: Geometric Posture Evaluation (new)
            logger.info("Phase 3c: Running geometric heuristics...")
            posture_errors = self.heuristics_engine.evaluate(
                action_segments, skeletons
            )
            results["posture_errors"] = posture_errors
            if posture_errors:
                logger.info("Detected %d posture errors", len(posture_errors))
            
            # Phase 4: Pattern Analysis
            logger.info("Phase 4: Analyzing patterns...")
            for class_idx, confidence in classifications:
                self.pattern_analyzer.add_classification(class_idx, confidence)
            
            results["statistics"] = self.pattern_analyzer.get_statistics_summary()
            self.current_bout_stats = results["statistics"]
            
            logger.info("Video processing complete")
            
        except Exception as e:
            logger.error(f"Error processing video: {e}")
            raise
        
        return results
    
    def _extract_pose_sequence(
        self,
        video_path: str
    ) -> Tuple[List[Dict[str, Tuple[float, float]]], Dict[str, Any]]:
        """Extract classifier skeletons using Target Tracking (M4) and Gatekeeper (M5)."""
        import cv2
        from ..inference.target_tracker import TargetTracker
        from ..inference.activity_gatekeeper import ActivityGatekeeper

        skeleton_extractor = getattr(self.pose_estimator, "extract_video_skeleton", None)
        if (
            callable(skeleton_extractor)
            and getattr(skeleton_extractor, "__func__", None) is not PoseEstimator.extract_video_skeleton
        ):
            skeletons = list(skeleton_extractor(video_path) or [])
            tracking_frames = [
                self.fencer_tracker.build_frame(
                    frame_index,
                    [self.fencer_tracker.candidate_from_skeleton(skeleton)]
                )
                for frame_index, skeleton in enumerate(skeletons)
            ]
            payload = self.fencer_tracker.build_payload(tracking_frames)
            payload["active_to_video_map"] = list(range(len(skeletons)))
            payload["locked_track_id"] = None
            payload["target_side"] = self.target_side
            payload["weapon_hand"] = self.weapon_hand
            payload["source_fps"] = 30.0
            return skeletons, payload

        tracker = TargetTracker(
            target_side=self.target_side,
            weapon_hand=self.weapon_hand,
        )
        
        skeletons = []
        tracking_frames = []
        active_to_video_map = []
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        source_fps = cap.get(cv2.CAP_PROP_FPS)
        if not source_fps or source_fps <= 0:
            source_fps = 30.0
        gatekeeper = ActivityGatekeeper(fps=max(1, int(round(source_fps))))

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            should_extract = (
                True
                if self.pose_estimator.backend == "mock"
                else gatekeeper.should_extract_pose()
            )
            if not should_extract:
                frame_count += 1
                continue
            
            detections = self.pose_estimator.extract_frame_fencers(frame, persist_track=True)
            target_skel, opp_skel = tracker.process_frame_detections(detections, frame_count)
            
            is_active = gatekeeper.update(
                target_skeleton=target_skel, 
                opponent_skeleton=opp_skel, 
                frame_width=frame.shape[1], 
                target_side=self.target_side
            )
            if self.pose_estimator.backend == "mock":
                # Mock mode is for structural smoke tests, not gatekeeper tuning.
                is_active = target_skel is not None
            
            if is_active and target_skel is not None:
                if self.pose_estimator.validate_skeleton(target_skel):
                    skeletons.append(target_skel)
                    active_to_video_map.append(frame_count)
                    
            frame_data = self.fencer_tracker.build_frame(frame_count, detections)
            frame_data["gatekeeper_state"] = gatekeeper.state
            knee_angle = gatekeeper._get_knee_angle(target_skel, self.target_side) if target_skel else 180.0
            frame_data["knee_angle"] = knee_angle
            
            tracking_frames.append(frame_data)
            
            frame_count += 1
            if frame_count % 100 == 0:
                logger.info(f"Processed {frame_count} frames")
                
        cap.release()
        logger.info(f"Extraction complete. Active skeleton frames: {len(skeletons)}/{frame_count}")
        payload = self.fencer_tracker.build_payload(tracking_frames)
        payload["active_to_video_map"] = active_to_video_map
        payload["locked_track_id"] = tracker.locked_track_id
        payload["target_side"] = self.target_side
        payload["weapon_hand"] = self.weapon_hand
        payload["source_fps"] = float(source_fps)
        return skeletons, payload

    def _run_inference(
        self,
        skeleton_array: np.ndarray,
        batch_process: bool = True,
        batch_size: int = 32
    ) -> List[Tuple[int, float]]:
        """
        Run model inference on skeleton array.
        
        Args:
            skeleton_array: Array of shape (num_frames, num_joints, 2)
            batch_process: Whether to use batching
            batch_size: Batch size for processing
            
        Returns:
            List of (class_idx, confidence) tuples
        """
        self._validate_inference_array(skeleton_array)
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        classifications = []
        num_frames = skeleton_array.shape[0]
        actual_channels = skeleton_array.shape[1] * skeleton_array.shape[2]
        if actual_channels != self.model_input_channels:
            raise ValueError(
                "Skeleton array channel mismatch: "
                f"expected {self.model_input_channels}, got {actual_channels}"
            )
        
        # Sliding window inference
        window_size = self.INFERENCE_WINDOW_SIZE
        stride = self.INFERENCE_STRIDE

        windows = [
            skeleton_array[start_idx:start_idx + window_size]
            for start_idx in range(0, num_frames - window_size + 1, stride)
        ]
        if not windows:
            return classifications

        with torch.no_grad():
            if batch_process:
                batches = [
                    windows[start_idx:start_idx + batch_size]
                    for start_idx in range(0, len(windows), batch_size)
                ]
            else:
                batches = [[window] for window in windows]

            for batch_windows in batches:
                # Prepare input tensor:
                # (batch, 28, 9, 2) -> (batch, 28, 18) -> (batch, 18, 28)
                batch_array = np.stack(batch_windows, axis=0)
                flat_windows = batch_array.reshape(len(batch_windows), window_size, -1)
                input_tensor = torch.from_numpy(flat_windows).float()
                input_tensor = input_tensor.permute(0, 2, 1)
                input_tensor = input_tensor.to(self.device)
                
                # Forward pass
                logits = self.model(input_tensor)
                probabilities = torch.softmax(logits, dim=1)

                pred_classes = torch.argmax(probabilities, dim=1)
                confidences = probabilities.gather(
                    1,
                    pred_classes.unsqueeze(1)
                ).squeeze(1)

                classifications.extend(
                    (int(pred_class), float(confidence))
                    for pred_class, confidence in zip(
                        pred_classes.cpu(),
                        confidences.cpu()
                    )
                )
        
        return classifications

    def _validate_inference_array(self, skeleton_array: np.ndarray):
        """Validate Phase 3 inference input shape and values."""
        if not isinstance(skeleton_array, np.ndarray):
            raise TypeError("skeleton_array must be a numpy array")
        if skeleton_array.ndim != 3 or skeleton_array.shape[2] != 2:
            raise ValueError(
                "skeleton_array must have shape (num_frames, num_joints, 2)"
            )
        if skeleton_array.shape[0] == 0:
            raise ValueError("skeleton_array must contain at least one frame")
        if skeleton_array.shape[1] == 0:
            raise ValueError("skeleton_array must contain at least one joint")
        if not np.all(np.isfinite(skeleton_array)):
            raise ValueError("skeleton_array contains non-finite values")

    def _refine_step_segments_with_tracking(
        self,
        action_segments: List[Dict[str, Any]],
        tracking_payload: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Reconcile final SF/SB labels with target motion in tracked video frames.

        This is especially helpful for oldstyle debugging because the final HUD
        is judged against the actual pixel-space motion that the user sees.
        """
        frames_meta = tracking_payload.get("frames", [])
        frames_by_index = {
            int(frame.get("frame_index", index)): frame
            for index, frame in enumerate(frames_meta)
        }
        if not frames_by_index:
            return action_segments

        corrections = 0
        refined_segments: List[Dict[str, Any]] = []
        for segment in action_segments:
            updated = dict(segment)
            if updated.get("action") not in self.STEP_ACTIONS:
                refined_segments.append(updated)
                continue

            motion = self._tracking_motion_for_segment(updated, frames_by_index)
            if motion is None:
                refined_segments.append(updated)
                continue

            updated["tracking_motion_dx_px"] = round(motion["dx_px"], 1)
            updated["tracking_motion_threshold_px"] = round(motion["threshold_px"], 1)
            updated["tracking_motion_corrected"] = False
            if motion["action"] != updated["action"]:
                updated["action"] = motion["action"]
                updated["tracking_motion_corrected"] = True
                corrections += 1

            refined_segments.append(updated)

        if corrections:
            logger.info(
                "Tracking-motion refinement corrected %d SF/SB segments.",
                corrections,
            )
        return refined_segments

    def _tracking_motion_for_segment(
        self,
        segment: Dict[str, Any],
        frames_by_index: Dict[int, Dict[str, Any]],
    ) -> Optional[Dict[str, float | str]]:
        """Infer SF/SB from target displacement across the mapped video segment."""
        start_frame = int(segment.get("video_start_frame", -1))
        end_frame = int(segment.get("video_end_frame", -1))
        if start_frame < 0 or end_frame < start_frame:
            return None

        start_track = self._target_track_for_frame(frames_by_index.get(start_frame))
        end_track = self._target_track_for_frame(frames_by_index.get(end_frame))
        if start_track is None or end_track is None:
            return None

        start_x = self._track_anchor_x(start_track)
        end_x = self._track_anchor_x(end_track)
        if start_x is None or end_x is None:
            return None

        threshold_px = self._track_motion_threshold_px(start_track, end_track)
        dx_px = float(end_x - start_x)
        signed_forward_motion = dx_px if self.target_side == "left" else -dx_px
        if abs(signed_forward_motion) < threshold_px:
            return None

        return {
            "dx_px": dx_px,
            "threshold_px": threshold_px,
            "action": "SF" if signed_forward_motion > 0 else "SB",
        }

    def _target_track_for_frame(
        self,
        frame_info: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """Return the tracked target-side fencer from one frame payload."""
        if not frame_info:
            return None
        for track in frame_info.get("tracks", []):
            if track.get("side") == self.target_side:
                return track
        return None

    @staticmethod
    def _track_anchor_x(track: Dict[str, Any]) -> Optional[float]:
        """Use front ankle when available, otherwise fall back to bbox center."""
        skeleton = track.get("skeleton") or {}
        front_ankle = skeleton.get("front_ankle")
        if isinstance(front_ankle, (list, tuple)) and len(front_ankle) == 2:
            return float(front_ankle[0])
        center = track.get("center")
        if isinstance(center, (list, tuple)) and len(center) == 2:
            return float(center[0])
        return None

    def _track_motion_threshold_px(
        self,
        start_track: Dict[str, Any],
        end_track: Dict[str, Any],
    ) -> float:
        """Scale the motion threshold to the observed target size when possible."""
        heights = []
        for track in (start_track, end_track):
            bbox = track.get("bbox")
            if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                heights.append(abs(float(bbox[3]) - float(bbox[1])))
        if heights:
            return max(self.STEP_TRACKING_MOTION_MIN_PX, 0.05 * (sum(heights) / len(heights)))
        return self.STEP_TRACKING_MOTION_MIN_PX

    def _load_model_checkpoint(self):
        """Load optional model weights and record why loading did or did not happen."""
        if not self.model_checkpoint_path:
            return

        checkpoint_path = Path(self.model_checkpoint_path)
        if not checkpoint_path.exists():
            self.model_checkpoint_error = (
                f"Model checkpoint not found: {self.model_checkpoint_path}"
            )
            logger.warning(self.model_checkpoint_error)
            return

        try:
            checkpoint = torch.load(self.model_checkpoint_path, map_location=self.device)
            state_dict, metadata = self._extract_checkpoint_payload(checkpoint)
            self._validate_checkpoint_metadata(metadata)
            try:
                self.model.load_state_dict(state_dict)
            except Exception:
                legacy_model = self._try_load_legacy_fencenet_checkpoint(state_dict)
                if legacy_model is None:
                    raise
                self.model = legacy_model
        except Exception as exc:
            self.model_checkpoint_error = str(exc)
            logger.warning(f"Could not load checkpoint: {exc}")
            return

        self.model_checkpoint_loaded = True
        self.model_checkpoint_metadata = metadata
        self.model_checkpoint_error = None
        logger.info(f"Loaded model checkpoint: {self.model_checkpoint_path}")

    def _try_load_legacy_fencenet_checkpoint(
        self,
        state_dict: Dict[str, Any],
    ) -> Optional[torch.nn.Module]:
        """Load a legacy FenceNet checkpoint when the current model is FenceNetV2."""
        if self.use_bifencenet:
            return None
        if not self._looks_like_legacy_fencenet_state_dict(state_dict):
            return None

        legacy_model = FenceNet(
            input_channels=self.model_input_channels,
            device=self.device,
        ).to(self.device)
        legacy_model.load_state_dict(state_dict)
        logger.info(
            "Loaded legacy FenceNet checkpoint into compatibility model: %s",
            self.model_checkpoint_path,
        )
        return legacy_model

    def _extract_checkpoint_payload(
        self,
        checkpoint: Any
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Extract a state_dict and optional metadata from supported checkpoint shapes."""
        if not isinstance(checkpoint, dict):
            raise TypeError(
                "Checkpoint must be a state_dict or a dict containing 'state_dict'"
            )

        metadata = self._extract_checkpoint_metadata(checkpoint)

        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif self._looks_like_state_dict(checkpoint):
            state_dict = checkpoint
        else:
            raise ValueError(
                "Checkpoint must contain 'state_dict' or 'model_state_dict'"
            )

        if not isinstance(state_dict, dict):
            raise TypeError("Checkpoint state_dict must be a dictionary")

        return state_dict, metadata

    def _extract_checkpoint_metadata(self, checkpoint: Dict[str, Any]) -> Dict[str, Any]:
        """Collect supported checkpoint metadata from nested or top-level fields."""
        metadata_keys = {
            "format_version",
            "model_type",
            "input_channels",
            "num_classes",
            "action_classes",
        }
        metadata = {
            key: checkpoint[key]
            for key in metadata_keys
            if key in checkpoint
        }

        nested_metadata = checkpoint.get("metadata")
        if nested_metadata is not None:
            if not isinstance(nested_metadata, dict):
                raise TypeError("Checkpoint metadata must be a dictionary")
            metadata.update(nested_metadata)

        return metadata

    @staticmethod
    def _looks_like_state_dict(checkpoint: Dict[str, Any]) -> bool:
        """Return whether a dict appears to be a plain PyTorch state_dict."""
        return bool(checkpoint) and all(
            hasattr(value, "shape")
            for value in checkpoint.values()
        )

    @staticmethod
    def _looks_like_legacy_fencenet_state_dict(state_dict: Dict[str, Any]) -> bool:
        """Return whether a state_dict matches the older FenceNet architecture."""
        keys = set(state_dict.keys())
        return (
            any(key.startswith("tcn_blocks.") for key in keys)
            and "fc1.weight" in keys
            and "fc2.weight" in keys
        )

    def _validate_checkpoint_metadata(self, metadata: Dict[str, Any]):
        """Validate optional checkpoint metadata against this pipeline instance."""
        expected_model_type = "bifencenet" if self.use_bifencenet else "fencenet"
        expected_action_classes = self.model.get_class_names()

        format_version = metadata.get("format_version")
        if (
            format_version is not None
            and int(format_version) != self.SUPPORTED_CHECKPOINT_FORMAT_VERSION
        ):
            raise ValueError(
                "Unsupported checkpoint format_version: "
                f"{format_version}. Expected "
                f"{self.SUPPORTED_CHECKPOINT_FORMAT_VERSION}."
            )

        model_type = metadata.get("model_type")
        if (
            model_type is not None
            and str(model_type).lower() != expected_model_type
        ):
            raise ValueError(
                "Checkpoint model_type mismatch: "
                f"expected {expected_model_type}, got {model_type}"
            )

        input_channels = metadata.get("input_channels")
        if (
            input_channels is not None
            and int(input_channels) != self.model_input_channels
        ):
            raise ValueError(
                "Checkpoint input_channels mismatch: "
                f"expected {self.model_input_channels}, got {input_channels}"
            )

        num_classes = metadata.get("num_classes")
        if num_classes is not None and int(num_classes) != self.model.NUM_CLASSES:
            raise ValueError(
                "Checkpoint num_classes mismatch: "
                f"expected {self.model.NUM_CLASSES}, got {num_classes}"
            )

        action_classes = metadata.get("action_classes")
        if (
            action_classes is not None
            and list(action_classes) != expected_action_classes
        ):
            raise ValueError(
                "Checkpoint action_classes mismatch: "
                f"expected {expected_action_classes}, got {action_classes}"
            )

    def get_runtime_metadata(self) -> Dict[str, Any]:
        """Return JSON-friendly metadata about the active pipeline backends."""
        pose_model = self.pose_estimator.model_path
        if pose_model is None and self.pose_estimator.backend == "ultralytics":
            pose_model = self.pose_estimator.DEFAULT_MODEL_PATH

        return {
            "device": self.device,
            "model_type": "bifencenet" if self.use_bifencenet else "fencenet",
            "model_input_channels": self.model_input_channels,
            "model_checkpoint": self.model_checkpoint_path,
            "model_checkpoint_loaded": self.model_checkpoint_loaded,
            "model_checkpoint_error": self.model_checkpoint_error,
            "model_checkpoint_metadata": self.model_checkpoint_metadata,
            "model_weights": (
                "checkpoint" if self.model_checkpoint_loaded else "random"
            ),
            "pose_backend": self.pose_estimator.backend,
            "pose_requested_backend": self.pose_estimator.requested_backend,
            "pose_model": pose_model,
            "tracking_strategy": self.fencer_tracker.TRACKING_STRATEGY,
            "tracking_identity_persistence": self.fencer_tracker.IDENTITY_NOTE,
        }

    def get_model_status(self) -> Dict[str, Any]:
        """Return JSON-friendly status for action-recognition model weights."""
        return {
            "model_type": "bifencenet" if self.use_bifencenet else "fencenet",
            "model_input_channels": self.model_input_channels,
            "model_checkpoint": self.model_checkpoint_path,
            "model_checkpoint_loaded": self.model_checkpoint_loaded,
            "model_checkpoint_error": self.model_checkpoint_error,
            "model_checkpoint_metadata": self.model_checkpoint_metadata,
            "model_weights": (
                "checkpoint" if self.model_checkpoint_loaded else "random"
            ),
        }
    
    def get_immediate_feedback(self) -> str:
        """Get immediate feedback during bout."""
        if not self.current_fencer_id:
            return "Set fencer ID first"

        stats = self.pattern_analyzer.get_statistics_summary()
        
        return self.coach_engine.generate_immediate_feedback(
            fencer_id=self.current_fencer_id,
            current_score=self.current_score,
            stats=stats,
            recent_actions=list(self.pattern_analyzer.action_history)
        )
    
    def get_break_strategy(self) -> str:
        """Get strategy advice during break."""
        if not self.current_fencer_id:
            return "Set fencer ID first"

        stats = self.pattern_analyzer.get_statistics_summary()
        
        return self.coach_engine.generate_break_strategy(
            fencer_id=self.current_fencer_id,
            opponent_id=self.current_opponent_id,
            current_score=self.current_score,
            fencer_stats=stats
        )
    
    def get_conclusive_feedback(self, bout_result: str) -> str:
        """Get post-bout feedback."""
        if not self.current_fencer_id:
            return "Set fencer ID first"
        
        bout_stats = self.pattern_analyzer.get_statistics_summary()
        
        # Save to profile
        try:
            self.profile_manager.save_bout(
                fencer_id=self.current_fencer_id,
                bout_data=bout_stats,
                opponent_id=self.current_opponent_id,
                result=bout_result
            )
        except Exception as e:
            logger.error(f"Error saving bout: {e}")
        
        return self.coach_engine.generate_conclusive_feedback(
            fencer_id=self.current_fencer_id,
            bout_result=bout_result,
            bout_statistics=bout_stats,
            opponent_id=self.current_opponent_id
        )
    
    def set_fencer(self, fencer_id: str, fencer_name: str = ""):
        """Set current fencer."""
        self.current_fencer_id = fencer_id
        
        # Create profile if doesn't exist
        if not self.profile_manager.load_profile(fencer_id):
            self.profile_manager.create_profile(
                fencer_id=fencer_id,
                name=fencer_name or fencer_id
            )
        
        logger.info(f"Fencer set: {fencer_id}")
    
    def set_opponent(self, opponent_id: str, opponent_name: str = ""):
        """Set current opponent."""
        self.current_opponent_id = opponent_id
        logger.info(f"Opponent set: {opponent_id}")
    
    def update_score(self, player_score: int, opponent_score: int):
        """Update current bout score."""
        self.current_score = {
            "player": player_score,
            "opponent": opponent_score
        }
        logger.info(f"Score updated: {player_score} - {opponent_score}")
    
    def reset_bout(self):
        """Reset for new bout."""
        self.pattern_analyzer.clear_history()
        self.current_bout_stats = {}
        self.current_score = {"player": 0, "opponent": 0}
        logger.info("Bout reset")
