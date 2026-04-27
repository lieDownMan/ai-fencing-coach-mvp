"""
Geometric Heuristics Engine — Posture evaluator for fencing actions.

Applies rule-based geometric checks to raw YOLO skeleton coordinates
within the time-windows identified by the Action Spotter (sliding window).

Spec reference: fixing_app.md § Module 2 + Module 4 (target_side)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Front-limb mapping by target_side (spec Module 4 rule update)
# ---------------------------------------------------------------------------
# If target_side == "left", the fencer faces right → front leg = right,
#                                                     front arm = right.
# If target_side == "right", the fencer faces left → front leg = left,
#                                                      front arm = left.
FRONT_LIMBS = {
    "left": {
        "hip": "right_hip",
        "knee": "right_knee",
        "ankle": "right_ankle",
        "wrist": "front_wrist",        # PoseEstimator already maps this
    },
    "right": {
        "hip": "left_hip",
        "knee": "left_knee",
        "ankle": "left_ankle",
        "wrist": "front_wrist",
    },
}

# Actions that trigger each rule
LUNGE_ACTIONS = {"R", "IS", "WW", "JS"}
ALL_ACTIONS = {"R", "IS", "WW", "JS", "SF", "SB"}
STEP_ACTIONS = {"SF", "SB"}


# ---------------------------------------------------------------------------
# Math utility
# ---------------------------------------------------------------------------

def calc_angle(
    a: np.ndarray, b: np.ndarray, c: np.ndarray
) -> float:
    """
    Calculate the angle ∠ABC in degrees, where B is the vertex.

    Given three 2-D joint coordinates A, B, C:

        θ = arccos( (BA · BC) / (|BA| × |BC|) ) × 180/π

    Parameters
    ----------
    a, b, c : np.ndarray
        Each of shape ``(2,)`` — (x, y) coordinates.

    Returns
    -------
    float
        Angle in degrees.  Returns 180.0 if vectors are degenerate.
    """
    ba = a - b
    bc = c - b
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)

    if norm_ba < 1e-8 or norm_bc < 1e-8:
        return 180.0  # degenerate, treat as straight

    cos_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
    cos_angle = float(np.clip(cos_angle, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_angle)))


# ---------------------------------------------------------------------------
# Individual rule checkers
# ---------------------------------------------------------------------------

def _get_joint(skeleton: Dict[str, Any], name: str) -> Optional[np.ndarray]:
    """Safely extract a joint (x, y) as a numpy array."""
    coord = skeleton.get(name)
    if coord is None:
        return None
    arr = np.asarray(coord, dtype=float)
    if arr.shape != (2,):
        return None
    if not np.all(np.isfinite(arr)):
        return None
    return arr


def _pelvis_center(skeleton: Dict[str, Any]) -> Optional[np.ndarray]:
    """Average of left_hip and right_hip."""
    lh = _get_joint(skeleton, "left_hip")
    rh = _get_joint(skeleton, "right_hip")
    if lh is None or rh is None:
        return None
    return (lh + rh) / 2.0


def check_lunge_overextension(
    skeletons: List[Dict[str, Any]],
    target_side: str = "left",
) -> Optional[Dict[str, Any]]:
    """
    Rule A — Lunge Over-extension (Knee over toes).

    Find the frame with maximum front-ankle displacement (peak lunge),
    then check the front-leg angle at [Hip, Knee, Ankle].

    Parameters
    ----------
    skeletons : list[dict]
        Skeleton dicts for the action window.
    target_side : str
        ``"left"`` or ``"right"`` — which side of the screen the target
        fencer stands on.  Determines which leg is the front leg.

    Returns an error dict or None.
    """
    if not skeletons:
        return None

    limbs = FRONT_LIMBS.get(target_side, FRONT_LIMBS["left"])
    hip_key = limbs["hip"]
    knee_key = limbs["knee"]
    ankle_key = limbs["ankle"]

    # Find frame of maximum displacement (front ankle x-displacement from frame 0)
    ref_ankle = _get_joint(skeletons[0], ankle_key)
    if ref_ankle is None:
        return None

    max_disp = 0.0
    peak_idx = 0
    for idx, skel in enumerate(skeletons):
        ankle = _get_joint(skel, ankle_key)
        if ankle is None:
            continue
        disp = float(np.linalg.norm(ankle - ref_ankle))
        if disp > max_disp:
            max_disp = disp
            peak_idx = idx

    peak = skeletons[peak_idx]
    hip = _get_joint(peak, hip_key)
    knee = _get_joint(peak, knee_key)
    ankle = _get_joint(peak, ankle_key)

    if hip is None or knee is None or ankle is None:
        return None

    angle = calc_angle(hip, knee, ankle)
    if angle < 90.0:
        return {
            "error": "Knee over toes",
            "severity": "high",
            "detail": f"Front knee angle {angle:.1f}° < 90° at peak extension (frame {peak_idx})",
            "angle": round(angle, 1),
            "frame": peak_idx,
        }
    return None


def check_guard_dropped(
    skeletons: List[Dict[str, Any]],
    target_side: str = "left",
) -> Optional[Dict[str, Any]]:
    """
    Rule B — Weapon Hand Height (Guard dropped).

    Check if the front wrist Y-coordinate drops below the pelvis center
    Y-coordinate in any frame.  (Y axis increases downward.)

    Parameters
    ----------
    skeletons : list[dict]
        Skeleton dicts for the action window.
    target_side : str
        ``"left"`` or ``"right"``.

    Returns an error dict or None.
    """
    limbs = FRONT_LIMBS.get(target_side, FRONT_LIMBS["left"])
    wrist_key = limbs["wrist"]

    worst_frame = None
    worst_diff = 0.0

    for idx, skel in enumerate(skeletons):
        wrist = _get_joint(skel, wrist_key)
        pelvis = _pelvis_center(skel)
        if wrist is None or pelvis is None:
            continue
        # Y increases downward → wrist_y > pelvis_y means guard is dropped
        diff = wrist[1] - pelvis[1]
        if diff > 0 and diff > worst_diff:
            worst_diff = diff
            worst_frame = idx

    if worst_frame is not None:
        return {
            "error": "Guard dropped",
            "severity": "medium",
            "detail": f"Weapon hand dropped below waist at frame {worst_frame}",
            "frame": worst_frame,
        }
    return None


def check_center_of_mass_bounce(
    skeletons: List[Dict[str, Any]],
    target_side: str = "left",
) -> Optional[Dict[str, Any]]:
    """
    Rule C — Center of Mass Bouncing.

    Calculate the vertical variance of the pelvis center across the window.
    If it exceeds 10 % of the bounding-box height, flag it.

    Parameters
    ----------
    skeletons : list[dict]
        Skeleton dicts for the action window.
    target_side : str
        ``"left"`` or ``"right"`` (unused currently but kept for API consistency).

    Returns an error dict or None.
    """
    pelvis_ys: List[float] = []
    all_ys: List[float] = []

    for skel in skeletons:
        pc = _pelvis_center(skel)
        if pc is not None:
            pelvis_ys.append(float(pc[1]))
        # Collect all joint Y values for bbox height estimate
        for joint_name in skel:
            coord = _get_joint(skel, joint_name)
            if coord is not None:
                all_ys.append(float(coord[1]))

    if len(pelvis_ys) < 5 or len(all_ys) < 2:
        return None

    bbox_height = max(all_ys) - min(all_ys)
    if bbox_height < 1e-6:
        return None

    pelvis_range = max(pelvis_ys) - min(pelvis_ys)
    variance_ratio = pelvis_range / bbox_height

    if variance_ratio > 0.10:
        return {
            "error": "Unstable center of mass",
            "severity": "medium",
            "detail": (
                f"Pelvis vertical variance {variance_ratio:.0%} of bbox height "
                f"(threshold: 10%)"
            ),
            "variance_ratio": round(variance_ratio, 3),
        }
    return None


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

class HeuristicsEngine:
    """
    Dispatches geometric posture checks based on the identified action label.

    Parameters
    ----------
    target_side : str
        ``"left"`` or ``"right"`` — which side of the screen the target
        fencer stands on.  Passed through to every rule checker so it
        knows which limbs are "front".

    Usage::

        engine = HeuristicsEngine(target_side="left")
        errors = engine.evaluate(action_segments, raw_skeletons)
    """

    def __init__(self, target_side: str = "left"):
        if target_side not in ("left", "right"):
            raise ValueError(f"target_side must be 'left' or 'right', got '{target_side}'")
        self.target_side = target_side

    def evaluate(
        self,
        action_segments: List[Dict[str, Any]],
        raw_skeletons: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Run posture checks on each detected action segment.

        Parameters
        ----------
        action_segments : list[dict]
            Output of ``SlidingWindowInference.run()``.  Each dict has
            ``start_frame``, ``end_frame``, ``action``, ``confidence``.
        raw_skeletons : list[dict]
            Frame-by-frame raw YOLO skeleton dicts (un-normalized, with
            original pixel coordinates).

        Returns
        -------
        list[dict]
            Posture error records.  Each contains ``error``, ``severity``,
            ``detail``, ``action``, ``segment_index``, and optional
            rule-specific fields.
        """
        errors: List[Dict[str, Any]] = []
        total_frames = len(raw_skeletons)

        for seg_idx, seg in enumerate(action_segments):
            action = seg["action"]
            start = seg["start_frame"]
            end = min(seg["end_frame"], total_frames)

            if start >= total_frames or start >= end:
                continue

            window_skeletons = raw_skeletons[start:end]
            if not window_skeletons:
                continue

            # Rule A: Lunge over-extension (attacking actions)
            if action in LUNGE_ACTIONS:
                err = check_lunge_overextension(window_skeletons, self.target_side)
                if err:
                    err["action"] = action
                    err["segment_index"] = seg_idx
                    err["start_frame"] = start
                    err["end_frame"] = end
                    errors.append(err)

            # Rule B: Guard dropped (all actions)
            if action in ALL_ACTIONS:
                err = check_guard_dropped(window_skeletons, self.target_side)
                if err:
                    err["action"] = action
                    err["segment_index"] = seg_idx
                    err["start_frame"] = start
                    err["end_frame"] = end
                    errors.append(err)

            # Rule C: Center of mass bouncing (step actions)
            if action in STEP_ACTIONS:
                err = check_center_of_mass_bounce(window_skeletons, self.target_side)
                if err:
                    err["action"] = action
                    err["segment_index"] = seg_idx
                    err["start_frame"] = start
                    err["end_frame"] = end
                    errors.append(err)

        logger.info(
            "Heuristics evaluated %d segments → %d posture errors",
            len(action_segments),
            len(errors),
        )
        return errors
