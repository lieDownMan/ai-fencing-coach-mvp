"""
Target Isolation & Tracking
Spec reference: fixing_app.md § Module 4
"""

import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger(__name__)

class TargetTracker:
    """
    Isolates the target fencer's skeleton stream using YOLOv8 ByteTrack.
    """
    
    def __init__(self, target_side: str = "left"):
        """
        Args:
            target_side: "left" or "right"
        """
        self.target_side = target_side
        self.locked_track_id: Optional[int] = None
        
        # Buffer for interpolation (max gap = 5)
        self.last_known_skeleton: Optional[Dict[str, Tuple[float, float]]] = None
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
            
        # Frame 0 logic: lock onto track_id based on target_side
        if self.locked_track_id is None and frame_idx == 0:
            if self.target_side == "left":
                # Lock track_id with minimum X
                target = min(valid_detections, key=lambda d: self._get_bbox_center_x(d["bbox"]))
            else:
                # Lock track_id with maximum X (could filter out refs if needed)
                target = max(valid_detections, key=lambda d: self._get_bbox_center_x(d["bbox"]))
                
            self.locked_track_id = target["track_id"]
            logger.info(f"Frame 0: Locked onto track_id {self.locked_track_id} as {self.target_side} fencer.")
        
        # Find the target and opponent
        target_det = None
        opponent_det = None
        
        for det in valid_detections:
            if det["track_id"] == self.locked_track_id:
                target_det = det
            else:
                # Naive opponent association: taking largest remaining or just any
                # In 2-person bout, the other is usually the opponent
                if opponent_det is None or det.get("area", 0) > opponent_det.get("area", 0):
                    opponent_det = det

        # If locked target is found
        if target_det is not None:
            self.last_known_skeleton = target_det["skeleton"]
            self.missing_frames_count = 0
            
            opp_skel = opponent_det["skeleton"] if opponent_det else None
            return target_det["skeleton"], opp_skel
            
        # Target missing, use padding/interpolation
        return self._handle_missing_target(), (opponent_det["skeleton"] if opponent_det else None)

    def _handle_missing_target(self) -> Optional[Dict[str, Any]]:
        """Pad missing frames with the last known skeleton if <= 5 frames."""
        if self.last_known_skeleton is not None and self.missing_frames_count < self.max_missing_frames:
            self.missing_frames_count += 1
            return self.last_known_skeleton
        return None
