"""Activity state machine for gating FenceNet inference."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


class ActivityGatekeeper:
    """Track whether the target fencer is actively fencing."""

    STATE_IDLE = "IDLE"
    STATE_CHECKING = "CHECKING"
    STATE_ACTIVE = "ACTIVE"

    def __init__(
        self,
        fps: int = 30,
        active_knee_angle_deg: float = 170.0,
        idle_knee_angle_deg: float = 174.0,
        motion_threshold_px: float = 1.5,
    ):
        self.fps = int(fps)
        self.state = self.STATE_IDLE
        self.frame_count = 0
        self.active_knee_angle_deg = float(active_knee_angle_deg)
        self.idle_knee_angle_deg = float(idle_knee_angle_deg)
        self.motion_threshold_px = float(motion_threshold_px)

        self.active_trigger_count = 0
        self.active_trigger_threshold = 5
        self.idle_trigger_count = 0
        self.idle_trigger_threshold = 2 * self.fps

        self.last_pelvis_center: Optional[np.ndarray] = None
        self.last_reasons: Dict[str, Any] = {}

    def should_extract_pose(self) -> bool:
        """Run pose at low FPS while idle and full FPS while checking/active."""
        self.frame_count += 1
        if self.state == self.STATE_IDLE:
            skip_rate = max(1, self.fps // 5)
            return (self.frame_count % skip_rate) == 1
        return True

    def _get_knee_angle(
        self,
        skeleton: Dict[str, Any],
        target_side: str,
    ) -> Optional[float]:
        from .heuristics_engine import FRONT_LIMBS, _get_joint, calc_angle

        limbs = FRONT_LIMBS.get(target_side, FRONT_LIMBS["left"])
        hip = _get_joint(skeleton, limbs["hip"])
        knee = _get_joint(skeleton, limbs["knee"])
        ankle = _get_joint(skeleton, limbs["ankle"])
        if hip is None or knee is None or ankle is None:
            return None
        return calc_angle(hip, knee, ankle)

    def _get_shoulder_width(self, skeleton: Dict[str, Any]) -> Optional[float]:
        from .heuristics_engine import _get_joint

        ls = _get_joint(skeleton, "left_shoulder")
        rs = _get_joint(skeleton, "right_shoulder")
        if ls is None or rs is None:
            return 100.0
        return float(np.linalg.norm(ls - rs))

    def _check_fencer_distance(
        self,
        target_skeleton: Dict[str, Any],
        opponent_skeleton: Optional[Dict[str, Any]],
        frame_width: int,
    ) -> bool:
        if opponent_skeleton is None:
            return False

        from .heuristics_engine import _pelvis_center

        target_center = _pelvis_center(target_skeleton)
        opponent_center = _pelvis_center(opponent_skeleton)
        if target_center is None or opponent_center is None:
            return False

        distance = abs(target_center[0] - opponent_center[0])
        return bool(distance > (frame_width * 0.6))

    def update(
        self,
        target_skeleton: Optional[Dict[str, Any]],
        opponent_skeleton: Optional[Dict[str, Any]],
        frame_width: int,
        target_side: str,
    ) -> bool:
        """Update the state machine and return whether the frame is active."""
        if target_skeleton is None:
            if self.state == self.STATE_ACTIVE:
                self.idle_trigger_count += 1
                if self.idle_trigger_count >= self.idle_trigger_threshold:
                    self.state = self.STATE_IDLE
                    self.idle_trigger_count = 0
            elif self.state == self.STATE_CHECKING:
                self.state = self.STATE_IDLE
                self.active_trigger_count = 0

            self.last_reasons = {
                "has_target": False,
                "state": self.state,
                "reason": "missing_target",
                "active_trigger_count": self.active_trigger_count,
                "idle_trigger_count": self.idle_trigger_count,
            }
            return self.state == self.STATE_ACTIVE

        knee_angle = self._get_knee_angle(target_skeleton, target_side)
        if knee_angle is None:
            knee_angle = 180.0

        shoulder_width = self._get_shoulder_width(target_skeleton)
        is_turned_back = shoulder_width < (frame_width * 0.05)
        too_far = self._check_fencer_distance(
            target_skeleton,
            opponent_skeleton,
            frame_width,
        )

        from .heuristics_engine import _pelvis_center

        pelvis_center = _pelvis_center(target_skeleton)
        pelvis_motion = 0.0
        if pelvis_center is not None and self.last_pelvis_center is not None:
            pelvis_motion = float(np.linalg.norm(pelvis_center - self.last_pelvis_center))
        moving = self.last_pelvis_center is None or pelvis_motion >= self.motion_threshold_px
        if pelvis_center is not None:
            self.last_pelvis_center = pelvis_center

        en_garde_posture = knee_angle < self.active_knee_angle_deg
        en_garde = en_garde_posture and (
            moving
            or self.state != self.STATE_IDLE
            or self.active_trigger_count > 0
        )
        standing_up = knee_angle > self.idle_knee_angle_deg
        stop_condition = standing_up or is_turned_back or too_far

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
                self.idle_trigger_count = 0

        self.last_reasons = {
            "has_target": True,
            "state": self.state,
            "knee_angle": float(knee_angle),
            "active_knee_angle_deg": self.active_knee_angle_deg,
            "idle_knee_angle_deg": self.idle_knee_angle_deg,
            "en_garde": bool(en_garde),
            "en_garde_posture": bool(en_garde_posture),
            "standing_up": bool(standing_up),
            "shoulder_width": float(shoulder_width) if shoulder_width is not None else None,
            "turned_back": bool(is_turned_back),
            "too_far": bool(too_far),
            "pelvis_motion": float(pelvis_motion),
            "moving": bool(moving),
            "active_trigger_count": self.active_trigger_count,
            "idle_trigger_count": self.idle_trigger_count,
        }
        return self.state == self.STATE_ACTIVE
