"""Helpers for canonicalizing fencing front-limb joints."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np


_BILATERAL_CHAIN = ("shoulder", "elbow", "wrist", "hip", "knee", "ankle")


def _as_point(value: Any) -> Optional[Tuple[float, float]]:
    """Return a finite (x, y) tuple when the value looks like a 2D point."""
    if value is None:
        return None
    point = np.asarray(value, dtype=float)
    if point.shape != (2,) or not np.all(np.isfinite(point)):
        return None
    return float(point[0]), float(point[1])


def _points_close(a: Any, b: Any, atol: float = 1e-4) -> bool:
    """Return True when two points represent the same 2D location."""
    point_a = _as_point(a)
    point_b = _as_point(b)
    if point_a is None or point_b is None:
        return False
    return bool(np.allclose(point_a, point_b, atol=atol))


def _fallback_front_side(screen_side: Optional[str]) -> str:
    """Return the legacy front-limb assumption for a screen-side label."""
    if screen_side == "right":
        return "left"
    return "right"


def infer_front_side(
    skeleton: Dict[str, Any],
    screen_side: Optional[str] = None,
    prefer_explicit_front: bool = False,
) -> str:
    """
    Infer whether the fencer's leading side is anatomical left or right.

    We first trust an already-canonical front ankle/wrist when available. When
    the skeleton still carries raw left/right joints, we infer the leading side
    from the average horizontal position of bilateral joints relative to the
    fencer's screen side.
    """
    left_ankle = skeleton.get("left_ankle")
    right_ankle = skeleton.get("right_ankle")
    front_ankle = skeleton.get("front_ankle")
    left_wrist = skeleton.get("left_wrist")
    right_wrist = skeleton.get("right_wrist")
    front_wrist = skeleton.get("front_wrist")

    if prefer_explicit_front:
        if _points_close(front_ankle, left_ankle) and not _points_close(front_ankle, right_ankle):
            return "left"
        if _points_close(front_ankle, right_ankle) and not _points_close(front_ankle, left_ankle):
            return "right"
        if _points_close(front_wrist, left_wrist) and not _points_close(front_wrist, right_wrist):
            return "left"
        if _points_close(front_wrist, right_wrist) and not _points_close(front_wrist, left_wrist):
            return "right"

    if screen_side in {"left", "right"}:
        left_xs = []
        right_xs = []
        for joint_name in _BILATERAL_CHAIN:
            left_point = _as_point(skeleton.get(f"left_{joint_name}"))
            right_point = _as_point(skeleton.get(f"right_{joint_name}"))
            if left_point is None or right_point is None:
                continue
            left_xs.append(left_point[0])
            right_xs.append(right_point[0])

        if left_xs and right_xs:
            left_mean_x = float(sum(left_xs) / len(left_xs))
            right_mean_x = float(sum(right_xs) / len(right_xs))
            if abs(left_mean_x - right_mean_x) >= 1e-4:
                if screen_side == "left":
                    return "left" if left_mean_x > right_mean_x else "right"
                return "left" if left_mean_x < right_mean_x else "right"

    if _points_close(front_ankle, left_ankle) and not _points_close(front_ankle, right_ankle):
        return "left"
    if _points_close(front_ankle, right_ankle) and not _points_close(front_ankle, left_ankle):
        return "right"

    if _points_close(front_wrist, left_wrist) and not _points_close(front_wrist, right_wrist):
        return "left"
    if _points_close(front_wrist, right_wrist) and not _points_close(front_wrist, left_wrist):
        return "right"

    if screen_side not in {"left", "right"}:
        return _fallback_front_side(screen_side)
    return _fallback_front_side(screen_side)


def front_limb_keys(
    skeleton: Dict[str, Any],
    screen_side: Optional[str] = None,
    prefer_explicit_front: bool = False,
) -> Dict[str, str]:
    """Return the anatomical joint names for the fencer's leading arm and leg."""
    front_side = infer_front_side(
        skeleton,
        screen_side=screen_side,
        prefer_explicit_front=prefer_explicit_front,
    )
    return {
        "side": front_side,
        "shoulder": f"{front_side}_shoulder",
        "elbow": f"{front_side}_elbow",
        "wrist": f"{front_side}_wrist",
        "hip": f"{front_side}_hip",
        "knee": f"{front_side}_knee",
        "ankle": f"{front_side}_ankle",
    }


def canonicalize_front_joints(
    skeleton: Dict[str, Any],
    screen_side: Optional[str] = None,
) -> Dict[str, Any]:
    """Copy a skeleton and rewrite front_* joints to the actual leading side."""
    canonical = dict(skeleton)
    keys = front_limb_keys(
        canonical,
        screen_side=screen_side,
        prefer_explicit_front=False,
    )

    for front_name, side_name in (
        ("front_shoulder", keys["shoulder"]),
        ("front_elbow", keys["elbow"]),
        ("front_wrist", keys["wrist"]),
        ("front_ankle", keys["ankle"]),
    ):
        point = _as_point(canonical.get(side_name))
        if point is not None:
            canonical[front_name] = point

    return canonical
