"""
Spatial Normalizer - Normalize skeleton coordinates.
- Subtract nose position from all joints in the first frame
- Divide by vertical distance between head and front ankle in first frame
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class SpatialNormalizer:
    """
    Normalizes skeleton coordinates by centering and scaling.
    """

    MODEL_JOINT_NAMES = [
        "nose",
        "front_wrist",
        "front_elbow",
        "front_shoulder",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
    ]
    
    def __init__(self):
        """Initialize the spatial normalizer."""
        self.reference_nose = None
        self.scale_factor = None
    
    def fit(self, skeleton_sequence: List[Dict[str, Tuple[float, float]]]):
        """
        Fit normalizer using first frame as reference.
        
        Args:
            skeleton_sequence: List of skeleton dictionaries
        """
        if not skeleton_sequence:
            raise ValueError("Empty skeleton sequence provided")
        
        first_frame = skeleton_sequence[0]
        
        # Get reference nose position
        if "nose" not in first_frame:
            raise KeyError("'nose' not found in first frame skeleton")
        
        self.reference_nose = np.array(first_frame["nose"])
        
        # Calculate scale factor: vertical distance between nose and front ankle
        if "front_ankle" not in first_frame:
            raise KeyError("'front_ankle' not found in first frame skeleton")
        
        head_pos = np.array(first_frame["nose"])
        ankle_pos = np.array(first_frame["front_ankle"])
        vertical_distance = abs(ankle_pos[1] - head_pos[1])
        
        if vertical_distance < 1e-6:
            logger.warning("Vertical distance is too small, using 1.0 as scale factor")
            self.scale_factor = 1.0
        else:
            self.scale_factor = vertical_distance
        
        logger.info(f"Normalizer fitted: reference_nose={self.reference_nose}, scale={self.scale_factor}")
    
    def normalize_skeleton(self, skeleton: Dict[str, Tuple[float, float]]) -> Dict[str, Tuple[float, float]]:
        """
        Normalize a single skeleton frame.
        
        Args:
            skeleton: Skeleton dictionary with (x, y) coordinates
            
        Returns:
            Normalized skeleton dictionary
        """
        if self.reference_nose is None or self.scale_factor is None:
            raise RuntimeError("Normalizer not fitted. Call fit() first.")
        
        normalized = {}
        for joint_name, (x, y) in skeleton.items():
            # Subtract reference nose position
            x_norm = (x - self.reference_nose[0]) / self.scale_factor
            y_norm = (y - self.reference_nose[1]) / self.scale_factor
            normalized[joint_name] = (x_norm, y_norm)
        
        return normalized
    
    def normalize_sequence(self, skeleton_sequence: List[Dict[str, Tuple[float, float]]]) -> List[Dict[str, Tuple[float, float]]]:
        """
        Normalize entire skeleton sequence.
        
        Args:
            skeleton_sequence: List of skeleton dictionaries
            
        Returns:
            List of normalized skeleton dictionaries
        """
        return [self.normalize_skeleton(skeleton) for skeleton in skeleton_sequence]
    
    def get_normalized_array(
        self,
        skeleton_sequence: List[Dict[str, Tuple[float, float]]],
        joint_names: Optional[List[str]] = None,
        already_normalized: bool = False
    ) -> np.ndarray:
        """
        Convert normalized skeleton sequence to numpy array.
        
        Args:
            skeleton_sequence: List of skeleton dictionaries
            joint_names: Optional explicit joint order to export
            already_normalized: Set to True if the input coordinates are already normalized
            
        Returns:
            np.ndarray of shape (num_frames, num_joints, 2)
        """
        if not skeleton_sequence:
            raise ValueError("Empty skeleton sequence")

        normalized_seq = (
            skeleton_sequence
            if already_normalized
            else self.normalize_sequence(skeleton_sequence)
        )
        joint_names = joint_names or sorted(normalized_seq[0].keys())
        num_frames = len(normalized_seq)
        num_joints = len(joint_names)
        
        array = np.zeros((num_frames, num_joints, 2))
        
        for frame_idx, frame in enumerate(normalized_seq):
            for joint_idx, joint_name in enumerate(joint_names):
                if joint_name not in frame:
                    raise KeyError(f"'{joint_name}' not found in frame {frame_idx}")
                array[frame_idx, joint_idx] = frame[joint_name]
        
        return array
