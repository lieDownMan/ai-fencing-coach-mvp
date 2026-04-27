"""
Activity Gatekeeper — State Machine (Idle vs. Active)

Spec reference: fixing_app.md § Module 5
"""

import logging
from typing import Dict, Any, Tuple, Optional
import numpy as np

logger = logging.getLogger(__name__)

class ActivityGatekeeper:
    """
    State machine that gates when to process frames and feed them to the Action Spotter.
    Helps prevent false positives during non-fencing periods (resting, walking).
    """

    STATE_IDLE = "IDLE"
    STATE_CHECKING = "CHECKING"  # Transitioning from IDLE to ACTIVE
    STATE_ACTIVE = "ACTIVE"

    def __init__(self, fps: int = 30):
        self.fps = fps
        self.state = self.STATE_IDLE
        self.frame_count = 0
        
        # State machine counters
        self.active_trigger_count = 0
        self.active_trigger_threshold = 5  # Permissive MVP: 5 frames instead of 15
        
        self.idle_trigger_count = 0
        self.idle_trigger_threshold = 2 * fps  # 2s cooldown
        
    def should_extract_pose(self) -> bool:
        """
        Determine if we should run YOLO on the current frame.
        IDLE: only 5 FPS (1 in 6 frames if 30 FPS).
        ACTIVE/CHECKING: full FPS.
        """
        self.frame_count += 1
        
        if self.state == self.STATE_IDLE:
            # Run at 5 FPS → 1 frame every 6 frames (assuming 30fps)
            skip_rate = max(1, self.fps // 5)
            # If skip_rate is 6, process frames 0, 6, 12...
            return (self.frame_count % skip_rate) == 1
        
        return True

    def _get_knee_angle(self, skeleton: Dict[str, Any], target_side: str) -> Optional[float]:
        """Calculate knee angle for the target side."""
        # Use calc_angle from heuristics_engine later, or calculate here
        from .heuristics_engine import calc_angle, FRONT_LIMBS, _get_joint
        
        limbs = FRONT_LIMBS.get(target_side, FRONT_LIMBS["left"])
        hip_key = limbs["hip"]
        knee_key = limbs["knee"]
        ankle_key = limbs["ankle"]
        
        hip = _get_joint(skeleton, hip_key)
        knee = _get_joint(skeleton, knee_key)
        ankle = _get_joint(skeleton, ankle_key)
        
        if hip is None or knee is None or ankle is None:
            return None
            
        return calc_angle(hip, knee, ankle)

    def _get_shoulder_width(self, skeleton: Dict[str, Any]) -> Optional[float]:
        """Calculate 2D distance between shoulders."""
        from .heuristics_engine import _get_joint
        ls = _get_joint(skeleton, "left_shoulder")
        rs = _get_joint(skeleton, "right_shoulder")
        if ls is None or rs is None:
            # Fallback to front_shoulder if available
            return 100.0  # Safe default if unavailable
        return float(np.linalg.norm(ls - rs))

    def _check_fencer_distance(
        self, 
        target_skeleton: Dict[str, Any], 
        opponent_skeleton: Optional[Dict[str, Any]], 
        frame_width: int
    ) -> bool:
        """Check if distance between fencers > 60% of frame width."""
        if opponent_skeleton is None:
            return False
            
        from .heuristics_engine import _pelvis_center
        tc = _pelvis_center(target_skeleton)
        oc = _pelvis_center(opponent_skeleton)
        
        if tc is None or oc is None:
            return False
            
        distance = abs(tc[0] - oc[0])
        return distance > (frame_width * 0.6)

    def update(
        self, 
        target_skeleton: Optional[Dict[str, Any]], 
        opponent_skeleton: Optional[Dict[str, Any]],
        frame_width: int,
        target_side: str
    ) -> bool:
        """
        Update state machine based on the current frame's skeletons.
        
        Returns:
            bool: True if the system is ACTIVE (meaning this frame should be kept for Sliding Window).
        """
        if target_skeleton is None:
            # If tracking lost, progress IDLE counter if active
            if self.state == self.STATE_ACTIVE:
                self.idle_trigger_count += 1
                if self.idle_trigger_count >= self.idle_trigger_threshold:
                    self.state = self.STATE_IDLE
                    self.idle_trigger_count = 0
            elif self.state == self.STATE_CHECKING:
                self.state = self.STATE_IDLE
                self.active_trigger_count = 0
            
            return self.state == self.STATE_ACTIVE

        knee_angle = self._get_knee_angle(target_skeleton, target_side)
        if knee_angle is None:
            knee_angle = 180.0  # default to standing if missing
            
        shoulder_width = self._get_shoulder_width(target_skeleton)
        is_turned_back = shoulder_width < (frame_width * 0.05)  # threshold for 2D projection width ~ 0
        
        too_far = self._check_fencer_distance(target_skeleton, opponent_skeleton, frame_width)
        
        # En Garde condition
        # M5 asked for 155, but 2D projection varies wildly; we use 170.0 for easier trigger
        en_garde = knee_angle < 170.0

        # Stop condition
        stop_condition = (knee_angle > 180.0) or is_turned_back or too_far

        if self.state == self.STATE_IDLE:
            if en_garde:
                self.state = self.STATE_CHECKING
                self.active_trigger_count = 1
                
        elif self.state == self.STATE_CHECKING:
            if en_garde:
                self.active_trigger_count += 1
                if self.active_trigger_count >= self.active_trigger_threshold:
                    self.state = self.STATE_ACTIVE
                    self.idle_trigger_count = 0
            else:
                self.state = self.STATE_IDLE
                self.active_trigger_count = 0
                
        elif self.state == self.STATE_ACTIVE:
            if stop_condition:
                self.idle_trigger_count += 1
                if self.idle_trigger_count >= self.idle_trigger_threshold:
                    self.state = self.STATE_IDLE
                    self.idle_trigger_count = 0
            else:
                # Reset cooldown if they go back into stance
                self.idle_trigger_count = 0

        return self.state == self.STATE_ACTIVE
