"""
Temporal Sampler - Resample skeleton sequences to fixed length.
Ensures exactly 28 frames in each sequence for consistent model input.
"""

import numpy as np
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


class TemporalSampler:
    """
    Resamples skeleton sequences to a fixed number of frames.
    """
    
    SEQUENCE_LENGTH = 28
    
    def __init__(self, target_length: int = SEQUENCE_LENGTH):
        """
        Initialize temporal sampler.
        
        Args:
            target_length: Target number of frames (default: 28)
        """
        if target_length <= 0:
            raise ValueError("target_length must be positive")
        self.target_length = target_length
    
    def sample(self, skeleton_sequence: List[Dict[str, Tuple[float, float]]]) -> List[Dict[str, Tuple[float, float]]]:
        """
        Resample skeleton sequence to target length.
        Uses linear interpolation between frames.
        
        Args:
            skeleton_sequence: Input skeleton sequence
            
        Returns:
            Resampled skeleton sequence of length target_length
        """
        num_frames = len(skeleton_sequence)
        
        if num_frames == 0:
            raise ValueError("Empty skeleton sequence")
        self._validate_consistent_keys(skeleton_sequence)

        if num_frames == self.target_length:
            return skeleton_sequence
        
        if num_frames < self.target_length:
            # Interpolate to increase number of frames
            sampled = self._interpolate_frames(skeleton_sequence)
        else:
            # Downsample to decrease number of frames
            sampled = self._downsample_frames(skeleton_sequence)
        
        return sampled
    
    def _interpolate_frames(self, skeleton_sequence: List[Dict[str, Tuple[float, float]]]) -> List[Dict[str, Tuple[float, float]]]:
        """
        Interpolate frames to increase sequence length.
        
        Args:
            skeleton_sequence: Input skeleton sequence
            
        Returns:
            Interpolated skeleton sequence
        """
        num_frames = len(skeleton_sequence)
        sampled = []
        
        # Calculate indices for target length
        indices = np.linspace(0, num_frames - 1, self.target_length)
        
        for idx in indices:
            if idx == int(idx):
                # Exact match
                sampled.append(skeleton_sequence[int(idx)])
            else:
                # Linear interpolation between two frames
                frame_idx = int(idx)
                alpha = idx - frame_idx
                
                frame1 = skeleton_sequence[frame_idx]
                frame2 = skeleton_sequence[frame_idx + 1]
                
                interpolated_frame = {}
                for joint_name in frame1.keys():
                    x1, y1 = frame1[joint_name]
                    x2, y2 = frame2[joint_name]
                    
                    x = x1 + alpha * (x2 - x1)
                    y = y1 + alpha * (y2 - y1)
                    interpolated_frame[joint_name] = (x, y)
                
                sampled.append(interpolated_frame)
        
        return sampled
    
    def _downsample_frames(self, skeleton_sequence: List[Dict[str, Tuple[float, float]]]) -> List[Dict[str, Tuple[float, float]]]:
        """
        Downsample frames to decrease sequence length.
        
        Args:
            skeleton_sequence: Input skeleton sequence
            
        Returns:
            Downsampled skeleton sequence
        """
        num_frames = len(skeleton_sequence)
        sampled = []
        
        # Calculate indices for target length
        indices = np.linspace(0, num_frames - 1, self.target_length, dtype=int)
        
        for idx in indices:
            sampled.append(skeleton_sequence[idx])
        
        return sampled
    
    def sample_array(self, skeleton_array: np.ndarray) -> np.ndarray:
        """
        Resample numpy array representation of skeleton sequence.
        
        Args:
            skeleton_array: np.ndarray of shape (num_frames, num_joints, 2)
            
        Returns:
            Resampled array of shape (target_length, num_joints, 2)
        """
        num_frames = skeleton_array.shape[0]
        if skeleton_array.ndim != 3 or skeleton_array.shape[2] != 2:
            raise ValueError(
                "skeleton_array must have shape (num_frames, num_joints, 2)"
            )
        if num_frames == 0:
            raise ValueError("Empty skeleton array")

        num_joints = skeleton_array.shape[1]
        
        if num_frames == self.target_length:
            return skeleton_array
        
        # Calculate indices for target length
        indices = np.linspace(0, num_frames - 1, self.target_length)
        
        resampled = np.zeros((self.target_length, num_joints, 2))
        
        for target_idx, source_idx in enumerate(indices):
            if source_idx == int(source_idx):
                resampled[target_idx] = skeleton_array[int(source_idx)]
            else:
                # Linear interpolation
                frame_idx = int(source_idx)
                alpha = source_idx - frame_idx
                
                frame1 = skeleton_array[frame_idx]
                frame2 = skeleton_array[frame_idx + 1]
                
                resampled[target_idx] = frame1 + alpha * (frame2 - frame1)
        
        return resampled

    @staticmethod
    def _validate_consistent_keys(
        skeleton_sequence: List[Dict[str, Tuple[float, float]]]
    ):
        """Ensure every frame exposes the same joints before interpolation."""
        expected_keys = set(skeleton_sequence[0].keys())
        for frame_idx, frame in enumerate(skeleton_sequence[1:], start=1):
            if set(frame.keys()) != expected_keys:
                raise KeyError(
                    f"Frame {frame_idx} has inconsistent skeleton joints"
                )
