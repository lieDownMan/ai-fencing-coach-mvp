"""
Target Isolation & Tracking
Spec reference: fixing_app.md § Module 4
"""

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
        self.locked_fallback_rank: Optional[int] = None
        
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
            detections: List of parsed candidate dicts from PoseEstimator.
                        ByteTrack ``track_id`` is preferred when available;
                        ``source_rank`` is used as a smoke-test fallback.
            frame_idx: Current frame number.
            
        Returns:
            Tuple of (target_skeleton_dict, opponent_skeleton_dict)
        """
        # Prefer ByteTrack IDs when available, but keep a source-rank fallback
        # so mock/non-tracked smoke runs can still exercise the live pipeline.
        valid_detections = [
            d for d in detections
            if d.get("skeleton") is not None and d.get("bbox") is not None
        ]
        
        if not valid_detections:
            return self._handle_missing_target(), None
            
        # Lock onto the first usable detection based on target_side. In a live
        # webcam run, frame 0 often has no track yet while the camera/model warm up.
        if self.locked_track_id is None and self.locked_fallback_rank is None:
            if self.target_side == "left":
                target = min(valid_detections, key=lambda d: self._get_bbox_center_x(d["bbox"]))
            else:
                target = max(valid_detections, key=lambda d: self._get_bbox_center_x(d["bbox"]))

            if target.get("track_id") is not None:
                self.locked_track_id = target["track_id"]
                logger.info(
                    "Frame %s: locked onto track_id %s as %s fencer.",
                    frame_idx,
                    self.locked_track_id,
                    self.target_side,
                )
            else:
                self.locked_fallback_rank = int(target.get("source_rank", 0))
                logger.info(
                    "Frame %s: locked onto source_rank %s as %s fencer.",
                    frame_idx,
                    self.locked_fallback_rank,
                    self.target_side,
                )
        
        # Find the target and opponent
        target_det = None
        opponent_det = None
        
        for det in valid_detections:
            is_locked_track = (
                self.locked_track_id is not None
                and det.get("track_id") == self.locked_track_id
            )
            is_locked_fallback = (
                self.locked_track_id is None
                and self.locked_fallback_rank is not None
                and int(det.get("source_rank", -1)) == self.locked_fallback_rank
            )

            if is_locked_track or is_locked_fallback:
                target_det = det
                if is_locked_fallback and det.get("track_id") is not None:
                    self.locked_track_id = det["track_id"]
                    self.locked_fallback_rank = None
                    logger.info(
                        "Promoted fallback target lock to track_id %s.",
                        self.locked_track_id,
                    )
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
