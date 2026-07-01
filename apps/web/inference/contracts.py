"""Shared data contracts for the Python coaching pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

Point = Tuple[float, float]
Skeleton = Dict[str, Point]


def _point(value: Any) -> Optional[Point]:
    try:
        if len(value) != 2:
            return None
        x = float(value[0])
        y = float(value[1])
    except (TypeError, ValueError):
        return None
    if not np.isfinite(x) or not np.isfinite(y):
        return None
    return x, y


def clean_skeleton(skeleton: Optional[Dict[str, Any]]) -> Skeleton:
    """Return a JSON-safe skeleton with only finite 2D points."""
    if not skeleton:
        return {}
    cleaned: Skeleton = {}
    for joint_name, coords in skeleton.items():
        point = _point(coords)
        if point is not None:
            cleaned[str(joint_name)] = point
    return cleaned


@dataclass
class PoseDetection:
    """One pose candidate with raw and canonical skeleton geometry."""

    bbox: Optional[List[float]]
    center: Optional[List[float]]
    area: float
    skeleton: Skeleton
    raw_skeleton: Skeleton = field(default_factory=dict)
    joint_confidence: Dict[str, float] = field(default_factory=dict)
    confidence: float = 1.0
    source_rank: int = 0
    track_id: Optional[int | str] = None
    interpolated: bool = False

    @classmethod
    def from_dict(cls, item: Dict[str, Any]) -> "PoseDetection":
        bbox = item.get("bbox")
        center = item.get("center")
        return cls(
            bbox=[float(v) for v in bbox] if bbox is not None else None,
            center=[float(v) for v in center] if center is not None else None,
            area=float(item.get("area") or 0.0),
            skeleton=clean_skeleton(item.get("skeleton")),
            raw_skeleton=clean_skeleton(item.get("raw_skeleton")),
            joint_confidence={
                str(key): float(value)
                for key, value in (item.get("joint_confidence") or {}).items()
            },
            confidence=float(item.get("confidence") or 0.0),
            source_rank=int(item.get("source_rank") or 0),
            track_id=item.get("track_id"),
            interpolated=bool(item.get("interpolated", False)),
        )

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "bbox": self.bbox,
            "center": self.center,
            "area": float(self.area),
            "skeleton": self.skeleton,
            "raw_skeleton": self.raw_skeleton,
            "joint_confidence": dict(self.joint_confidence),
            "confidence": float(self.confidence),
            "source_rank": int(self.source_rank),
            "interpolated": bool(self.interpolated),
        }
        if self.track_id is not None:
            data["track_id"] = self.track_id
        return data


@dataclass
class FrameSample:
    """One processed frame in the clip/realtime coaching pipeline."""

    frame_index: int
    timestamp: float
    tracks: List[Dict[str, Any]] = field(default_factory=list)
    target_detection: Optional[Dict[str, Any]] = None
    opponent_detection: Optional[Dict[str, Any]] = None
    target_skeleton: Skeleton = field(default_factory=dict)
    opponent_skeleton: Skeleton = field(default_factory=dict)
    gate_state: str = "IDLE"
    gate_reasons: Dict[str, Any] = field(default_factory=dict)
    normalized_skeleton: Optional[np.ndarray] = None
    valid_for_model: bool = False
    interpolated: bool = False
    active_index: Optional[int] = None

    def to_report_dict(self) -> Dict[str, Any]:
        return {
            "frame_index": int(self.frame_index),
            "timestamp": float(self.timestamp),
            "tracks": self.tracks,
            "target_detection": self.target_detection,
            "opponent_detection": self.opponent_detection,
            "target_skeleton": self.target_skeleton,
            "opponent_skeleton": self.opponent_skeleton,
            "target_bbox": (
                self.target_detection.get("bbox")
                if self.target_detection
                else None
            ),
            "target_track_id": (
                self.target_detection.get("track_id")
                if self.target_detection
                else None
            ),
            "gatekeeper_state": self.gate_state,
            "gate_reasons": self.gate_reasons,
            "knee_angle": self.gate_reasons.get("knee_angle"),
            "valid_for_model": bool(self.valid_for_model),
            "interpolated": bool(self.interpolated),
            "active_index": self.active_index,
        }
