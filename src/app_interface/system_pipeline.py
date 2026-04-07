"""
System Pipeline - Orchestrates the complete fencing coaching system.
Coordinates: Pose Estimation -> Preprocessing -> FenceNet -> Tracking -> LLM Coaching
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import logging

from ..pose_estimation import PoseEstimator
from ..preprocessing import SpatialNormalizer, TemporalSampler
from ..models import FenceNet, BiFenceNet
from ..tracking import PatternAnalyzer, ProfileManager
from ..llm_agent import CoachEngine

logger = logging.getLogger(__name__)


class SystemPipeline:
    """
    Main orchestrator for the fencing coaching system.
    Manages all pipeline stages from video input to coaching output.
    """

    SUPPORTED_CHECKPOINT_FORMAT_VERSION = 1
    
    def __init__(
        self,
        use_bifencenet: bool = False,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        model_checkpoint: Optional[str] = None,
        profiles_dir: str = "data/fencer_profiles/",
        pose_backend: str = "auto",
        pose_model_path: Optional[str] = None,
        llm_model_name: str = "llava-next"
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
        
        # Phase 2: Preprocessing
        self.spatial_normalizer = SpatialNormalizer()
        self.temporal_sampler = TemporalSampler(target_length=28)
        self.model_joint_names = list(SpatialNormalizer.MODEL_JOINT_NAMES)
        self.model_input_channels = len(self.model_joint_names) * 2
        
        # Phase 3: FenceNet Model
        if use_bifencenet:
            self.model = BiFenceNet(
                input_channels=self.model_input_channels,
                hidden_channels=64,
                num_tcn_blocks=6,
                device=device
            ).to(device)
            logger.info("Using BiFenceNet model")
        else:
            self.model = FenceNet(
                input_channels=self.model_input_channels,
                hidden_channels=64,
                num_tcn_blocks=6,
                device=device
            ).to(device)
            logger.info("Using FenceNet model")
        
        self._load_model_checkpoint()
        
        self.model.eval()
        
        # Phase 4: Pattern Tracking
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
            "statistics": {}
        }
        
        try:
            # Phase 1: Extract Skeleton
            logger.info("Phase 1: Extracting skeleton...")
            skeletons = self.pose_estimator.extract_video_skeleton(video_path)
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
            
            # Phase 3: Model Inference
            logger.info("Phase 3: Running FenceNet inference...")
            classifications = self._run_inference(skeleton_array, batch_process)
            results["classifications"] = classifications
            
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
        window_size = 28  # Match temporal sampler
        stride = 14  # 50% overlap

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
                # (batch, 28, 10, 2) -> (batch, 28, 20) -> (batch, 20, 28)
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
            self.model.load_state_dict(state_dict)
        except Exception as exc:
            self.model_checkpoint_error = str(exc)
            logger.warning(f"Could not load checkpoint: {exc}")
            return

        self.model_checkpoint_loaded = True
        self.model_checkpoint_metadata = metadata
        self.model_checkpoint_error = None
        logger.info(f"Loaded model checkpoint: {self.model_checkpoint_path}")

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
