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
    
    def __init__(
        self,
        use_bifencenet: bool = False,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        model_checkpoint: Optional[str] = None,
        profiles_dir: str = "data/fencer_profiles/",
        pose_backend: str = "auto",
        pose_model_path: Optional[str] = None
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
        """
        self.device = device
        self.use_bifencenet = use_bifencenet
        
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
        
        # Load checkpoint if provided
        if model_checkpoint and Path(model_checkpoint).exists():
            try:
                checkpoint = torch.load(model_checkpoint, map_location=device)
                self.model.load_state_dict(checkpoint)
                logger.info(f"Loaded model checkpoint: {model_checkpoint}")
            except Exception as e:
                logger.warning(f"Could not load checkpoint: {e}")
        
        self.model.eval()
        
        # Phase 4: Pattern Tracking
        self.pattern_analyzer = PatternAnalyzer()
        self.profile_manager = ProfileManager(profiles_dir=profiles_dir)
        
        # Phase 5: LLM Coaching
        self.coach_engine = CoachEngine(profiles_dir=profiles_dir, device=device)
        
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
        
        with torch.no_grad():
            for start_idx in range(0, num_frames - window_size + 1, stride):
                end_idx = start_idx + window_size
                window = skeleton_array[start_idx:end_idx]
                
                # Prepare input tensor
                # Reshape from (28, 10, 2) to (28, 20) then to (1, 20, 28)
                flat_window = window.reshape(window_size, -1)  # (28, 20)
                input_tensor = torch.from_numpy(flat_window).float()
                input_tensor = input_tensor.permute(1, 0).unsqueeze(0)  # (1, 20, 28)
                input_tensor = input_tensor.to(self.device)
                
                # Forward pass
                logits = self.model(input_tensor)
                probabilities = torch.softmax(logits, dim=1)[0]
                
                # Get prediction
                pred_class = torch.argmax(probabilities).item()
                confidence = probabilities[pred_class].item()
                
                classifications.append((pred_class, confidence))
        
        return classifications
    
    def get_immediate_feedback(self) -> str:
        """Get immediate feedback during bout."""
        if not self.current_fencer_id:
            return "Set fencer ID first"
        
        return self.coach_engine.generate_immediate_feedback(
            fencer_id=self.current_fencer_id,
            current_score=self.current_score
        )
    
    def get_break_strategy(self) -> str:
        """Get strategy advice during break."""
        if not self.current_fencer_id:
            return "Set fencer ID first"
        
        return self.coach_engine.generate_break_strategy(
            fencer_id=self.current_fencer_id,
            opponent_id=self.current_opponent_id,
            current_score=self.current_score
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
        self.current_score = {"player": 0, "opponent": 0}
        logger.info("Bout reset")
