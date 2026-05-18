"""
Geometric Heuristics Engine — Posture evaluator for fencing actions.

Returns machine-readable error *keys* (matching coach_playbook.json)
instead of human-readable strings so that downstream consumers
(video annotator, real-time voice coach, LLM agent) can each render
the appropriate representation.

Detectable keys (via skeleton geometry):
    bounce_excessive, lunge_overextension, guard_dropped,
    foot_before_hand, stance_too_high, incomplete_arm_extension,
    over_parrying, wide_step, narrow_step,
    center_of_mass_in_front, center_of_mass_leaning_backward

Keys reserved for future detection (require sword-tip / multi-segment):
    wide_disengage
"""

from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional
import numpy as np

logger = logging.getLogger(__name__)

FRONT_LIMBS = {
    "left": {
        "hip": "right_hip", "knee": "right_knee", "ankle": "right_ankle",
        "wrist": "front_wrist", "elbow": "front_elbow", "shoulder": "front_shoulder",
    },
    "right": {
        "hip": "left_hip", "knee": "left_knee", "ankle": "left_ankle",
        "wrist": "front_wrist", "elbow": "front_elbow", "shoulder": "front_shoulder",
    },
}

BACK_LIMBS = {
    "left": {
        "hip": "left_hip", "knee": "left_knee", "ankle": "left_ankle",
        "wrist": "back_wrist",
    },
    "right": {
        "hip": "right_hip", "knee": "right_knee", "ankle": "right_ankle",
        "wrist": "back_wrist",
    },
}

# ---------------------------------------------------------------------------
# Math utilities
# ---------------------------------------------------------------------------

def calc_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Return the angle ABC (vertex at B) in degrees."""
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


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class HeuristicsEngine:
    """Evaluate skeleton sequences and emit error keys from coach_playbook.json."""

    def __init__(self, target_side: str = "left", training_mode: str = "Free Bouting"):
        self.target_side = target_side
        self.training_mode = training_mode

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        action_segments: List[Dict[str, Any]],
        raw_skeletons: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Return a list of error dicts, each containing an ``error_key`` field."""
        errors: List[Dict[str, Any]] = []
        for seg_idx, seg in enumerate(action_segments):
            action = seg["action"]
            start = seg["start_frame"]
            end = min(seg["end_frame"], len(raw_skeletons))
            if start >= len(raw_skeletons) or start >= end:
                continue
            window_skeletons = raw_skeletons[start:end]

            seg_errors = self._check_rules(action, window_skeletons)
            for err in seg_errors:
                err.update({
                    "action": action,
                    "segment_index": seg_idx,
                    "start_frame": start,
                    "end_frame": end,
                })
                errors.append(err)
        return errors

    # ------------------------------------------------------------------
    # Rule dispatcher
    # ------------------------------------------------------------------

    def _check_rules(
        self, action: str, skeletons: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Run all applicable rules and return every triggered error."""
        if not skeletons:
            return []

        triggered: List[Dict[str, Any]] = []
        is_offensive = action in ["R", "JS", "WW", "IS"]

        # --- Footwork rules (Mode A, SF/SB) ---
        if self.training_mode == "Footwork" and action in ["SF", "SB"]:
            self._try_append(triggered, self._check_bounce(skeletons))
            self._try_append(triggered, self._check_stance_too_high(skeletons))
            self._try_append(triggered, self._check_step_width(skeletons))
            self._try_append(triggered, self._check_center_of_mass(skeletons))

        # --- Offensive lunge rules (Mode B, R/JS/WW/IS) ---
        if self.training_mode == "Target Practice" and is_offensive:
            self._try_append(triggered, self._check_lunge(skeletons))
            self._try_append(triggered, self._check_foot_before_hand(skeletons))
            self._try_append(triggered, self._check_incomplete_arm_extension(skeletons))

        # --- Guard Dropped (all modes, tolerance varies) ---
        self._try_append(triggered, self._check_guard(skeletons))

       
        # --- Stance too high + Bounce (en-garde check, all modes, neutral) ---
        if self.training_mode != "Footwork" and action in ["SF", "SB"]:
            self._try_append(triggered, self._check_stance_too_high(skeletons))
            self._try_append(triggered, self._check_bounce(skeletons))
            self._try_append(triggered, self._check_step_width(skeletons))
            self._try_append(triggered, self._check_center_of_mass(skeletons))

        # --- Over-parrying (defensive context: SB all modes, or SF/SB in Free Bouting) ---
        if action == "SB" or (self.training_mode == "Free Bouting" and action in ["SF", "SB"]):
            self._try_append(triggered, self._check_over_parrying(skeletons))

        return triggered

    @staticmethod
    def _try_append(
        target: List[Dict[str, Any]], err: Optional[Dict[str, Any]]
    ) -> None:
        if err is not None:
            target.append(err)

    # ------------------------------------------------------------------
    # Rule 1: bounce_excessive — 步伐上下浮動
    # Trigger: Footwork mode, SF/SB
    # ------------------------------------------------------------------

    def _check_bounce(
        self, skeletons: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        pelvis_ys: List[float] = []
        all_ys: List[float] = []
        for skel in skeletons:
            pc = _pelvis_center(skel)
            if pc is not None:
                pelvis_ys.append(float(pc[1]))
            for j in skel.values():
                if isinstance(j, (list, tuple, np.ndarray)) and len(j) == 2:
                    all_ys.append(float(j[1]))

        if len(pelvis_ys) < 5 or len(all_ys) < 2:
            return None
        bbox_height = max(all_ys) - min(all_ys)
        if bbox_height < 1e-6:
            return None

        delta_y = max(pelvis_ys) - min(pelvis_ys)
        if delta_y > 0.25 * bbox_height:
            return {"error_key": "bounce_excessive"}
        return None

    # ------------------------------------------------------------------
    # Rule 2: lunge_overextension — 弓步過度前傾
    # Trigger: Target Practice mode, R/JS/WW/IS
    # ------------------------------------------------------------------

    def _check_lunge(
        self, skeletons: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        limbs = FRONT_LIMBS[self.target_side]
        ref_ankle = _get_joint(skeletons[0], limbs["ankle"])
        if ref_ankle is None:
            return None

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
        if hip is None or knee is None or ankle is None:
            return None

        angle = calc_angle(hip, knee, ankle)
        if angle < 90.0:
            return {"error_key": "lunge_overextension"}
        return None

    # ------------------------------------------------------------------
    # Rule 3: guard_dropped — 持劍手掉落
    # Trigger: All modes (threshold varies)
    # ------------------------------------------------------------------

    def _check_guard(
        self, skeletons: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        limbs = FRONT_LIMBS[self.target_side]
        consecutive_frames = 0
        threshold = 20 if self.training_mode == "Free Bouting" else 10

        for skel in skeletons:
            wrist = _get_joint(skel, limbs["wrist"])
            pelvis = _pelvis_center(skel)
            if wrist is not None and pelvis is not None and wrist[1] > pelvis[1]:
                consecutive_frames += 1
                if consecutive_frames > threshold:
                    return {"error_key": "guard_dropped"}
            else:
                consecutive_frames = 0
        return None

    # ------------------------------------------------------------------
    # Rule 4: foot_before_hand — 手腳順序錯誤
    # Trigger: Target Practice, R/JS/WW/IS
    # Logic:  Compare frame index of peak wrist displacement vs. peak
    #         ankle displacement.  If ankle peaks first the foot moved
    #         before the hand.
    # ------------------------------------------------------------------

    def _check_foot_before_hand(
        self, skeletons: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        limbs = FRONT_LIMBS[self.target_side]
        ref_wrist = _get_joint(skeletons[0], limbs["wrist"])
        ref_ankle = _get_joint(skeletons[0], limbs["ankle"])
        if ref_wrist is None or ref_ankle is None:
            return None

        max_wrist_disp = 0.0
        wrist_peak_frame = 0
        max_ankle_disp = 0.0
        ankle_peak_frame = 0

        for i, skel in enumerate(skeletons):
            wrist = _get_joint(skel, limbs["wrist"])
            ankle = _get_joint(skel, limbs["ankle"])
            if wrist is not None:
                d = abs(float(wrist[0] - ref_wrist[0]))
                if d > max_wrist_disp:
                    max_wrist_disp = d
                    wrist_peak_frame = i
            if ankle is not None:
                d = abs(float(ankle[0] - ref_ankle[0]))
                if d > max_ankle_disp:
                    max_ankle_disp = d
                    ankle_peak_frame = i

        # Ankle peaks before wrist ⇒ foot moved first
        if (max_ankle_disp > 5 and max_wrist_disp > 5
                and ankle_peak_frame < wrist_peak_frame):
            return {"error_key": "foot_before_hand"}
        return None

    # ------------------------------------------------------------------
    # Rule 5: stance_too_high — 預備姿勢沒蹲好
    # Trigger: Footwork SF/SB or neutral stance in other modes
    # Logic:  Average front knee angle across the window.  If the knee
    #         stays near-straight (> 160°) the fencer is standing too
    #         tall.
    # ------------------------------------------------------------------

    def _check_stance_too_high(
        self, skeletons: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        limbs = FRONT_LIMBS[self.target_side]
        angles: List[float] = []
        for skel in skeletons:
            hip = _get_joint(skel, limbs["hip"])
            knee = _get_joint(skel, limbs["knee"])
            ankle = _get_joint(skel, limbs["ankle"])
            if hip is not None and knee is not None and ankle is not None:
                angles.append(calc_angle(hip, knee, ankle))

        if len(angles) < 3:
            return None

        avg_angle = float(np.mean(angles))
        # A proper en-garde has front knee ~120-140°.  > 160° means
        # the fencer is basically standing upright.
        if avg_angle > 170.0:
            return {"error_key": "stance_too_high"}
        return None

    # ------------------------------------------------------------------
    # Rule 6: incomplete_arm_extension — 刺的時候手沒有伸直
    # Trigger: Target Practice, R/JS/WW/IS
    # Logic:  At the frame of maximum wrist-forward displacement,
    #         measure the shoulder-elbow-wrist angle.  If < 155° the
    #         arm is not fully extended.
    # ------------------------------------------------------------------

    def _check_incomplete_arm_extension(
        self, skeletons: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        limbs = FRONT_LIMBS[self.target_side]
        ref_wrist = _get_joint(skeletons[0], limbs["wrist"])
        if ref_wrist is None:
            return None

        # Find the peak-extension frame (max wrist horizontal displacement)
        max_disp = 0.0
        peak_skel = skeletons[0]
        for skel in skeletons:
            wrist = _get_joint(skel, limbs["wrist"])
            if wrist is not None:
                d = abs(float(wrist[0] - ref_wrist[0]))
                if d > max_disp:
                    max_disp = d
                    peak_skel = skel

        shoulder = _get_joint(peak_skel, limbs["shoulder"])
        elbow = _get_joint(peak_skel, limbs["elbow"])
        wrist = _get_joint(peak_skel, limbs["wrist"])
        if shoulder is None or elbow is None or wrist is None:
            return None

        arm_angle = calc_angle(shoulder, elbow, wrist)
        if arm_angle < 155.0:
            return {"error_key": "incomplete_arm_extension"}
        return None

    # ------------------------------------------------------------------
    # Rule 9: over_parrying — 防守動作太大且太頻繁
    # Trigger: SB in all modes; SF/SB in Free Bouting
    # Logic:  Measure the horizontal sweep range of front_wrist across
    #         the window.  If the wrist travels more than 2× the
    #         shoulder width, the parry motion is excessively large.
    # ------------------------------------------------------------------

    def _check_over_parrying(
        self, skeletons: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        limbs = FRONT_LIMBS[self.target_side]

        # --- Determine shoulder width as a body-proportion reference ---
        shoulder_width = None
        for skel in skeletons:
            shoulder = _get_joint(skel, limbs["shoulder"])
            # Try to find the *other* shoulder for width measurement
            other_shoulder_name = (
                "left_shoulder" if self.target_side == "right" else "right_shoulder"
            )
            other_shoulder = _get_joint(skel, other_shoulder_name)
            if other_shoulder is None:
                # fallback: use front_shoulder ↔ pelvis center as proxy
                pelvis = _pelvis_center(skel)
                if shoulder is not None and pelvis is not None:
                    shoulder_width = abs(float(shoulder[0] - pelvis[0])) * 2.0
                    break
            else:
                if shoulder is not None:
                    shoulder_width = abs(float(shoulder[0] - other_shoulder[0]))
                    break

        if shoulder_width is None or shoulder_width < 1e-6:
            return None

        # --- Collect wrist X positions across the window ---
        wrist_xs: List[float] = []
        for skel in skeletons:
            wrist = _get_joint(skel, limbs["wrist"])
            if wrist is not None:
                wrist_xs.append(float(wrist[0]))

        if len(wrist_xs) < 5:
            return None

        sweep_range = max(wrist_xs) - min(wrist_xs)

        # Wrist sweeps more than 2× shoulder width ⇒ over-parrying
        if sweep_range > 2.0 * shoulder_width:
            return {"error_key": "over_parrying"}
        return None

    # ------------------------------------------------------------------
    # Rule 10: wide_step / narrow_step — 步伐太大/太小
    # Trigger: SF/SB in all modes
    # Logic: Measure the distance between front and back ankles compared
    #        to the shoulder width. If ratio > 3.0, wide. If < 1.0, narrow.
    # ------------------------------------------------------------------

    def _check_step_width(
        self, skeletons: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        limbs = FRONT_LIMBS[self.target_side]
        
        for skel in skeletons:
            front_ankle = _get_joint(skel, limbs["ankle"])
            back_ankle_name = "left_ankle" if self.target_side == "left" else "right_ankle"
            back_ankle = _get_joint(skel, back_ankle_name)
            
            front_shoulder = _get_joint(skel, limbs["shoulder"])
            pelvis = _pelvis_center(skel)
            if front_ankle is None or back_ankle is None or front_shoulder is None or pelvis is None:
                continue
                
            # Proxy for shoulder width: distance from front shoulder to pelvis center * 1.3
            shoulder_width = abs(front_shoulder[0] - pelvis[0]) * 2.5
            
            if shoulder_width < 10.0:
                continue
                
            step_width = abs(front_ankle[0] - back_ankle[0])
            ratio = step_width / shoulder_width
            
            if ratio > 3.0:
                return {"error_key": "wide_step"}
            elif ratio < 1.0:
                return {"error_key": "narrow_step"}
        return None

    # ------------------------------------------------------------------
    # Rule 11: center_of_mass_in_front / center_of_mass_leaning_backward
    # Trigger: SF/SB in all modes
    # Logic: Determine the pelvis x-position relative to the front and 
    #        back ankles. If ratio > 0.65 or < 0.35, the CoM is skewed.
    # ------------------------------------------------------------------

    def _check_center_of_mass(
        self, skeletons: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        limbs = FRONT_LIMBS[self.target_side]
        
        for skel in skeletons:
            front_ankle = _get_joint(skel, limbs["ankle"])
            back_ankle_name = "left_ankle" if self.target_side == "left" else "right_ankle"
            back_ankle = _get_joint(skel, back_ankle_name)
            pelvis = _pelvis_center(skel)
            
            if front_ankle is None or back_ankle is None or pelvis is None:
                continue
                
            front_x = float(front_ankle[0])
            back_x = float(back_ankle[0])
            pelvis_x = float(pelvis[0])
            
            base_width = abs(front_x - back_x)
            if base_width < 10.0:
                continue
                
            # 0.0 means pelvis is directly over back ankle
            # 1.0 means pelvis is directly over front ankle
            if front_x > back_x:
                ratio = (pelvis_x - back_x) / base_width
            else:
                ratio = (back_x - pelvis_x) / base_width
                
            if ratio > 0.65:
                return {"error_key": "center_of_mass_in_front"}
            elif ratio < 0.35:
                return {"error_key": "center_of_mass_leaning_backward"}
                
        return None