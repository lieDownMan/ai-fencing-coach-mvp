"""
Target Isolation & Tracking
Spec reference: fixing_app.md § Module 4
"""

import logging
import math
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
        self.locked_fallback_bbox: Optional[List[float]] = None
        
        # Buffer for interpolation (max gap = 5)
        self.last_known_skeleton: Optional[Dict[str, Tuple[float, float]]] = None
        self.last_known_bbox: Optional[List[float]] = None
        self.missing_frames_count = 0
        self.max_missing_frames = 5
        self.max_position_jump = 1.75
        self.max_track_jump = 2.5
        
    def _get_bbox_center_x(self, bbox: List[float]) -> float:
        """Calculate center X of a bounding box [x1, y1, x2, y2]."""
        return (bbox[0] + bbox[2]) / 2.0

    def _get_bbox_center(self, bbox: List[float]) -> Tuple[float, float]:
        """Calculate center point of a bounding box [x1, y1, x2, y2]."""
        return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)

    def _bbox_area(self, bbox: List[float]) -> float:
        return max(1.0, abs(float(bbox[2] - bbox[0]) * float(bbox[3] - bbox[1])))

    def _bbox_diag(self, bbox: List[float]) -> float:
        width = float(bbox[2] - bbox[0])
        height = float(bbox[3] - bbox[1])
        return max(1.0, math.hypot(width, height))

    def _position_score(self, det: Dict[str, Any], reference_bbox: List[float]) -> float:
        """Lower score means this detection is more likely the same fencer."""
        cx, cy = self._get_bbox_center(det["bbox"])
        ref_cx, ref_cy = self._get_bbox_center(reference_bbox)
        center_dist = math.hypot(cx - ref_cx, cy - ref_cy) / self._bbox_diag(reference_bbox)

        # Area is a weak secondary cue. It helps reject the opponent if one
        # fencer is much closer to the camera, but position remains dominant.
        area_ratio = abs(math.log(self._bbox_area(det["bbox"]) / self._bbox_area(reference_bbox)))
        return center_dist + 0.25 * area_ratio

    def _pick_initial_target(self, detections: List[Dict[str, Any]]) -> Dict[str, Any]:
        if self.target_side == "left":
            return min(detections, key=lambda d: self._get_bbox_center_x(d["bbox"]))
        return max(detections, key=lambda d: self._get_bbox_center_x(d["bbox"]))

    def _match_by_position(self, detections: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        reference_bbox = self.last_known_bbox or self.locked_fallback_bbox
        if reference_bbox is None:
            return None

        best = min(detections, key=lambda d: self._position_score(d, reference_bbox))
        score = self._position_score(best, reference_bbox)
        if score > self.max_position_jump:
            logger.debug(
                "Rejecting position fallback candidate with score %.2f > %.2f.",
                score,
                self.max_position_jump,
            )
            return None
        return best

    def _is_plausible_track_match(self, det: Dict[str, Any]) -> bool:
        if self.last_known_bbox is None:
            return True
        score = self._position_score(det, self.last_known_bbox)
        if score <= self.max_track_jump:
            return True
        logger.debug(
            "Rejecting locked track_id %s because it jumped %.2f > %.2f.",
            det.get("track_id"),
            score,
            self.max_track_jump,
        )
        return False

    def _remember_target(self, target_det: Dict[str, Any]) -> None:
        self.last_known_skeleton = target_det["skeleton"]
        self.last_known_bbox = list(target_det["bbox"])
        self.locked_fallback_bbox = list(target_det["bbox"])
        self.missing_frames_count = 0

    def reset(self) -> None:
        """Clear target identity so the next frame locks from target_side again."""
        self.locked_track_id = None
        self.locked_fallback_bbox = None
        self.last_known_skeleton = None
        self.last_known_bbox = None
        self.missing_frames_count = 0

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
                        bbox continuity is used as a fallback.
            frame_idx: Current frame number.
            
        Returns:
            Tuple of (target_skeleton_dict, opponent_skeleton_dict)
        """
        # Prefer ByteTrack IDs when available, but do not treat source_rank as
        # identity. Detector output order can change frame-to-frame, which makes
        # source_rank a common cause of left/right target flips.
        valid_detections = [
            d for d in detections
            if d.get("skeleton") is not None and d.get("bbox") is not None
        ]
        
        if not valid_detections:
            return self._handle_missing_target(), None
            
        # Lock onto the first usable detection based on target_side. In a live
        # webcam run, frame 0 often has no track yet while the camera/model warm up.
        if self.locked_track_id is None and self.locked_fallback_bbox is None:
            target = self._pick_initial_target(valid_detections)

            if target.get("track_id") is not None:
                self.locked_track_id = target["track_id"]
                logger.info(
                    "Frame %s: locked onto track_id %s as %s fencer.",
                    frame_idx,
                    self.locked_track_id,
                    self.target_side,
                )
            else:
                self.locked_fallback_bbox = list(target["bbox"])
                logger.info(
                    "Frame %s: locked onto %s fencer by position fallback.",
                    frame_idx,
                    self.target_side,
                )
        
        # Find the target and opponent
        target_det = None
        opponent_det = None
        
        if self.locked_track_id is not None:
            for det in valid_detections:
                if det.get("track_id") == self.locked_track_id:
                    if self._is_plausible_track_match(det):
                        target_det = det
                    break

        if target_det is None:
            target_det = self._match_by_position(valid_detections)
            if target_det is not None and target_det.get("track_id") is not None:
                old_track_id = self.locked_track_id
                self.locked_track_id = target_det["track_id"]
                logger.info(
                    "Frame %s: recovered target lock by position, track_id %s -> %s.",
                    frame_idx,
                    old_track_id,
                    self.locked_track_id,
                )

        for det in valid_detections:
            if det is target_det:
                target_det = det
            else:
                # Naive opponent association: taking largest remaining or just any
                # In 2-person bout, the other is usually the opponent
                if opponent_det is None or det.get("area", 0) > opponent_det.get("area", 0):
                    opponent_det = det

        # If locked target is found
        if target_det is not None:
            self._remember_target(target_det)
            
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
