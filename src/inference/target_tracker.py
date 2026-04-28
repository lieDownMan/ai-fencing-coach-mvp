"""
Target Isolation & Tracking
Spec reference: fixing_app.md § Module 4
"""

import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple

from ..fencing_skeleton import canonicalize_front_joints, normalize_weapon_hand

logger = logging.getLogger(__name__)

class TargetTracker:
    """
    Isolates the target fencer's skeleton stream using YOLOv8 ByteTrack.
    """
    
    def __init__(self, target_side: str = "left", weapon_hand: str = "auto"):
        """
        Args:
            target_side: "left" or "right"
        """
        self.target_side = target_side
        self.weapon_hand = normalize_weapon_hand(weapon_hand)
        self.locked_track_id: Optional[int] = None
        
        # Buffer for interpolation (max gap = 5)
        self.last_known_skeleton: Optional[Dict[str, Tuple[float, float]]] = None
        self.last_known_center: Optional[Tuple[float, float]] = None
        self.missing_frames_count = 0
        self.max_missing_frames = 5
        
    def _get_bbox_center_x(self, bbox: List[float]) -> float:
        """Calculate center X of a bounding box [x1, y1, x2, y2]."""
        return (bbox[0] + bbox[2]) / 2.0

    def process_frame_detections(
        self, 
        detections: List[Dict[str, Any]], 
        frame_idx: int
    ) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """
        Process trackers for one frame and return target and opponent skeletons.
        
        Args:
            detections: List of parsed candidate dicts from PoseEstimator, 
                        which MUST include "track_id" from model.track().
            frame_idx: Current frame number.
            
        Returns:
            Tuple of (target_skeleton_dict, opponent_skeleton_dict)
        """
        # Filter detections that have track_id
        valid_detections = [d for d in detections if d.get("track_id") is not None]
        
        if not valid_detections:
            return self._handle_missing_target(), None
        
        target_det = None
        if self.locked_track_id is None:
            target_det = self._select_side_candidate(valid_detections)
            self.locked_track_id = target_det["track_id"]
            logger.info(
                "Locked onto track_id %s as %s fencer at frame %d.",
                self.locked_track_id,
                self.target_side,
                frame_idx,
            )
        else:
            for det in valid_detections:
                if det["track_id"] == self.locked_track_id:
                    target_det = det
                    break

        if target_det is None:
            target_det = self._reacquire_target(valid_detections, frame_idx)

        opponent_det = self._select_opponent(valid_detections, target_det)
        opponent_side = "right" if self.target_side == "left" else "left"

        # If locked target is found
        if target_det is not None:
            target_skeleton = canonicalize_front_joints(
                target_det["skeleton"],
                screen_side=self.target_side,
                weapon_hand=self.weapon_hand,
            )
            self.last_known_skeleton = target_skeleton
            center = target_det.get("center")
            if center is not None and len(center) == 2:
                self.last_known_center = (float(center[0]), float(center[1]))
            self.missing_frames_count = 0

            opp_skel = (
                canonicalize_front_joints(
                    opponent_det["skeleton"],
                    screen_side=opponent_side,
                    weapon_hand="auto",
                )
                if opponent_det
                else None
            )
            return target_skeleton, opp_skel
            
        # Target missing, use padding/interpolation
        return self._handle_missing_target(), (
            canonicalize_front_joints(
                opponent_det["skeleton"],
                screen_side=opponent_side,
                weapon_hand="auto",
            )
            if opponent_det
            else None
        )

    def _handle_missing_target(self) -> Optional[Dict[str, Any]]:
        """Pad missing frames with the last known skeleton if <= 5 frames."""
        if self.last_known_skeleton is not None and self.missing_frames_count < self.max_missing_frames:
            self.missing_frames_count += 1
            return self.last_known_skeleton
        if self.locked_track_id is not None:
            logger.info(
                "Lost locked target track_id %s after %d missing frames; clearing lock.",
                self.locked_track_id,
                self.missing_frames_count,
            )
        self.locked_track_id = None
        self.last_known_skeleton = None
        self.last_known_center = None
        return None

    def _select_side_candidate(
        self,
        detections: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Pick the leftmost or rightmost candidate based on target_side."""
        if self.target_side == "left":
            return min(detections, key=lambda d: self._get_bbox_center_x(d["bbox"]))
        return max(detections, key=lambda d: self._get_bbox_center_x(d["bbox"]))

    def _reacquire_target(
        self,
        detections: List[Dict[str, Any]],
        frame_idx: int
    ) -> Optional[Dict[str, Any]]:
        """
        Reacquire the target when ByteTrack changes IDs or the original lock disappears.

        We first prefer the detection closest to the previous target center. If no
        prior center is available, we fall back to the configured side heuristic.
        """
        if not detections:
            return None

        if self.last_known_center is not None:
            candidate = min(
                detections,
                key=lambda det: self._center_distance(det.get("center")),
            )
        else:
            candidate = self._select_side_candidate(detections)

        previous_track_id = self.locked_track_id
        self.locked_track_id = candidate["track_id"]
        logger.info(
            "Reacquired target track: %s -> %s at frame %d.",
            previous_track_id,
            self.locked_track_id,
            frame_idx,
        )
        return candidate

    def _select_opponent(
        self,
        detections: List[Dict[str, Any]],
        target_det: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """Return the largest remaining detection that is not the target."""
        if target_det is None:
            return None

        opponent_det = None
        for det in detections:
            if det["track_id"] == target_det["track_id"]:
                continue
            if opponent_det is None or det.get("area", 0) > opponent_det.get("area", 0):
                opponent_det = det
        return opponent_det

    def _center_distance(self, center: Optional[List[float]]) -> float:
        """Return Euclidean distance from the last known target center."""
        if self.last_known_center is None or center is None or len(center) != 2:
            return float("inf")
        dx = float(center[0]) - self.last_known_center[0]
        dy = float(center[1]) - self.last_known_center[1]
        return float((dx * dx + dy * dy) ** 0.5)
