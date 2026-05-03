"""
Geometric Heuristics Engine — Posture evaluator for fencing actions.
"""

from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional
import numpy as np

logger = logging.getLogger(__name__)

FRONT_LIMBS = {
    "left": {"hip": "right_hip", "knee": "right_knee", "ankle": "right_ankle", "wrist": "front_wrist"},
    "right": {"hip": "left_hip", "knee": "left_knee", "ankle": "left_ankle", "wrist": "front_wrist"},
}

def calc_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ba = a - b
    bc = c - b
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    if norm_ba < 1e-8 or norm_bc < 1e-8:
        return 180.0
    cos_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
    cos_angle = float(np.clip(cos_angle, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_angle)))

def _get_joint(skeleton: Dict[str, Any], name: str) -> Optional[np.ndarray]:
    coord = skeleton.get(name)
    if coord is None:
        return None
    arr = np.asarray(coord, dtype=float)
    if arr.shape != (2,) or not np.all(np.isfinite(arr)):
        return None
    return arr

def _pelvis_center(skeleton: Dict[str, Any]) -> Optional[np.ndarray]:
    lh = _get_joint(skeleton, "left_hip")
    rh = _get_joint(skeleton, "right_hip")
    if lh is None or rh is None:
        return None
    return (lh + rh) / 2.0

class HeuristicsEngine:
    def __init__(self, target_side: str = "left", training_mode: str = "Free Bouting"):
        self.target_side = target_side
        self.training_mode = training_mode

    def evaluate(self, action_segments: List[Dict[str, Any]], raw_skeletons: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        errors = []
        for seg_idx, seg in enumerate(action_segments):
            action = seg["action"]
            start = seg["start_frame"]
            end = min(seg["end_frame"], len(raw_skeletons))
            if start >= len(raw_skeletons) or start >= end:
                continue
            window_skeletons = raw_skeletons[start:end]
            
            err = self._check_rules(action, window_skeletons)
            if err:
                err.update({"action": action, "segment_index": seg_idx, "start_frame": start, "end_frame": end})
                errors.append(err)
        return errors

    def _check_rules(self, action: str, skeletons: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not skeletons:
            return None
            
        # Rule 1: Center of Mass Bounce (Mode A, SF/SB)
        if self.training_mode == "Footwork" and action in ["SF", "SB"]:
            err = self._check_bounce(skeletons)
            if err: return err
            
        # Rule 2: Lunge Knee Over-extension (Mode B, R/JS/WW/IS)
        if self.training_mode == "Target Practice" and action in ["R", "JS", "WW", "IS"]:
            err = self._check_lunge(skeletons)
            if err: return err
            
        # Rule 3: Guard Dropped (All modes)
        err = self._check_guard(skeletons)
        if err: return err

        return None

    def _check_bounce(self, skeletons: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        pelvis_ys = []
        all_ys = []
        for skel in skeletons:
            pc = _pelvis_center(skel)
            if pc is not None: pelvis_ys.append(float(pc[1]))
            for j in skel.values():
                if len(j) == 2: all_ys.append(float(j[1]))
                
        if len(pelvis_ys) < 5 or len(all_ys) < 2: return None
        bbox_height = max(all_ys) - min(all_ys)
        if bbox_height < 1e-6: return None
        
        delta_y = max(pelvis_ys) - min(pelvis_ys)
        if delta_y > 0.1 * bbox_height:
            return {"error": "Warning: Excessive vertical bouncing during footwork."}
        return None

    def _check_lunge(self, skeletons: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        limbs = FRONT_LIMBS[self.target_side]
        ref_ankle = _get_joint(skeletons[0], limbs["ankle"])
        if ref_ankle is None: return None
        
        max_disp = 0.0
        peak_skel = skeletons[0]
        for skel in skeletons:
            ankle = _get_joint(skel, limbs["ankle"])
            if ankle is not None:
                disp = float(np.linalg.norm(ankle - ref_ankle))
                if disp > max_disp:
                    max_disp = disp
                    peak_skel = skel
                    
        hip = _get_joint(peak_skel, limbs["hip"])
        knee = _get_joint(peak_skel, limbs["knee"])
        ankle = _get_joint(peak_skel, limbs["ankle"])
        if hip is None or knee is None or ankle is None: return None
        
        angle = calc_angle(hip, knee, ankle)
        if angle < 90.0:
            return {"error": "Warning: Front knee angle is too acute. Over-lunging detected."}
        return None

    def _check_guard(self, skeletons: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        limbs = FRONT_LIMBS[self.target_side]
        consecutive_frames = 0
        threshold = 20 if self.training_mode == "Free Bouting" else 10
        
        for skel in skeletons:
            wrist = _get_joint(skel, limbs["wrist"])
            pelvis = _pelvis_center(skel)
            if wrist is not None and pelvis is not None and wrist[1] > pelvis[1]:
                consecutive_frames += 1
                if consecutive_frames > threshold:
                    return {"error": "Warning: Weapon hand dropped below waist, exposing valid target."}
            else:
                consecutive_frames = 0
        return None
