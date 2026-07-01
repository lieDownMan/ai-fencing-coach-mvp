"""
Two-fencer candidate tracking for side-view fencing video.

This tracker intentionally stays lightweight for the MVP: it keeps the two
largest pose candidates in each frame and assigns left/right labels by x-center.
It is useful for visualization and distance summaries, but it is not a robust
identity tracker through crossings or occlusion.
"""

from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


class FencerTracker:
    """Assign side-based fencer tracks from per-frame pose detections."""

    SCHEMA_VERSION = 1
    TRACKING_STRATEGY = "largest_two_candidates_sorted_by_center_x"
    IDENTITY_NOTE = (
        "Side-based labels; identities may swap if fencers cross or occlude."
    )
    TOO_CLOSE_DISTANCE_RATIO = 1.0
    TOO_CLOSE_RULE = (
        "too_close when front-ankle x-distance is less than 1.0 times "
        "average tracked fencer bbox height"
    )

    def build_frame(
        self,
        frame_index: int,
        detections: Iterable[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Build one JSON-friendly tracking frame from pose detections."""
        valid_detections = [
            self._normalize_detection(detection)
            for detection in detections
            if detection and detection.get("skeleton")
        ]
        valid_detections = [
            detection for detection in valid_detections if detection is not None
        ]
        selected = sorted(
            valid_detections,
            key=lambda detection: detection.get("area", 0.0),
            reverse=True
        )[:2]

        if len(selected) >= 2:
            selected = sorted(selected, key=lambda detection: detection["center"][0])
            labels = [("fencer_L", "left"), ("fencer_R", "right")]
        else:
            labels = [("fencer_1", "unknown")]

        tracks = [
            self._build_track(track_id, side, detection)
            for (track_id, side), detection in zip(labels, selected)
        ]
        engagement_distance, distance_source = self._engagement_distance(tracks)
        distance_features = self._distance_features(tracks, engagement_distance)

        return {
            "frame_index": int(frame_index),
            "detected_fencer_count": len(valid_detections),
            "tracked_fencer_count": len(tracks),
            "tracks": tracks,
            "engagement_distance_px": engagement_distance,
            "distance_source": distance_source,
            **distance_features,
        }

    def build_payload(self, frames: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Build a complete two-fencer tracking payload."""
        return {
            "schema_version": self.SCHEMA_VERSION,
            "strategy": self.TRACKING_STRATEGY,
            "identity_persistence": self.IDENTITY_NOTE,
            "too_close_rule": self.TOO_CLOSE_RULE,
            "summary": self.summarize(frames),
            "frames": frames,
        }

    def summarize(self, frames: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Summarize two-fencer coverage and engagement distance."""
        frame_count = len(frames)
        frames_with_two = sum(
            1 for frame in frames if frame.get("tracked_fencer_count", 0) >= 2
        )
        frames_with_one = sum(
            1 for frame in frames if frame.get("tracked_fencer_count", 0) == 1
        )
        frames_with_none = sum(
            1 for frame in frames if frame.get("tracked_fencer_count", 0) == 0
        )
        frames_too_close = sum(
            1 for frame in frames if frame.get("distance_status") == "too_close"
        )
        frames_distance_ok = sum(
            1 for frame in frames if frame.get("distance_status") == "ok"
        )
        distances = [
            float(frame["engagement_distance_px"])
            for frame in frames
            if frame.get("engagement_distance_px") is not None
        ]
        distance_ratios = [
            float(frame["engagement_distance_ratio"])
            for frame in frames
            if frame.get("engagement_distance_ratio") is not None
        ]
        average_distance = (
            float(sum(distances) / len(distances)) if distances else None
        )
        average_distance_ratio = (
            float(sum(distance_ratios) / len(distance_ratios))
            if distance_ratios
            else None
        )

        return {
            "frames_analyzed": frame_count,
            "frames_with_two_fencers": frames_with_two,
            "frames_with_one_fencer": frames_with_one,
            "frames_with_no_fencers": frames_with_none,
            "frames_too_close": frames_too_close,
            "frames_distance_ok": frames_distance_ok,
            "two_fencer_coverage": (
                float(frames_with_two / frame_count) if frame_count else 0.0
            ),
            "too_close_ratio": (
                float(frames_too_close / frame_count) if frame_count else 0.0
            ),
            "average_engagement_distance_px": average_distance,
            "average_engagement_distance_ratio": average_distance_ratio,
            "too_close_distance_ratio": self.TOO_CLOSE_DISTANCE_RATIO,
            "too_close_rule": self.TOO_CLOSE_RULE,
            "tracking_strategy": self.TRACKING_STRATEGY,
            "identity_persistence": self.IDENTITY_NOTE,
        }

    def candidate_from_skeleton(
        self,
        skeleton: Dict[str, Tuple[float, float]],
        confidence: float = 1.0,
        source_rank: int = 0
    ) -> Dict[str, Any]:
        """Build a detection candidate when only one skeleton is available."""
        bbox = self._bbox_from_skeleton(skeleton)
        center = self._center_from_bbox_or_skeleton(bbox, skeleton)
        area = self._bbox_area(bbox)
        return {
            "skeleton": skeleton,
            "bbox": bbox,
            "center": center,
            "area": area,
            "confidence": float(confidence),
            "source_rank": int(source_rank),
        }

    def _normalize_detection(
        self,
        detection: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        skeleton = detection.get("skeleton")
        if not isinstance(skeleton, dict):
            return None

        bbox = detection.get("bbox")
        if bbox is None:
            bbox = self._bbox_from_skeleton(skeleton)
        else:
            bbox = self._as_float_list(bbox, expected_len=4)

        center = detection.get("center")
        if center is None:
            center = self._center_from_bbox_or_skeleton(bbox, skeleton)
        else:
            center = self._as_float_list(center, expected_len=2)
        if center is None:
            return None

        area = detection.get("area")
        area = self._bbox_area(bbox) if area is None else float(area)
        confidence = float(detection.get("confidence", 0.0))
        source_rank = int(detection.get("source_rank", 0))
        track_id = detection.get("track_id")

        result = {
            "skeleton": skeleton,
            "bbox": bbox,
            "center": center,
            "area": area,
            "confidence": confidence,
            "source_rank": source_rank,
        }
        if track_id is not None:
            result["track_id"] = track_id
        return result

    def _build_track(
        self,
        fencer_label: str,
        side: str,
        detection: Dict[str, Any]
    ) -> Dict[str, Any]:
        result = {
            "fencer_label": fencer_label,
            "side": side,
            "source_rank": detection["source_rank"],
            "bbox": detection["bbox"],
            "center": detection["center"],
            "area": detection["area"],
            "confidence": detection["confidence"],
            "skeleton": self._json_skeleton(detection["skeleton"]),
        }
        if "track_id" in detection:
            result["track_id"] = detection["track_id"]
        return result

    def _engagement_distance(
        self,
        tracks: List[Dict[str, Any]]
    ) -> Tuple[Optional[float], Optional[str]]:
        if len(tracks) < 2:
            return None, None

        left_skeleton = tracks[0].get("skeleton", {})
        right_skeleton = tracks[1].get("skeleton", {})
        left_front_ankle = left_skeleton.get("front_ankle")
        right_front_ankle = right_skeleton.get("front_ankle")
        if left_front_ankle is not None and right_front_ankle is not None:
            return (
                abs(float(right_front_ankle[0]) - float(left_front_ankle[0])),
                "front_ankle_x",
            )

        return (
            abs(float(tracks[1]["center"][0]) - float(tracks[0]["center"][0])),
            "center_x",
        )

    def _distance_features(
        self,
        tracks: List[Dict[str, Any]],
        engagement_distance: Optional[float]
    ) -> Dict[str, Any]:
        """Build scale-relative distance feedback for one frame."""
        empty = {
            "average_fencer_height_px": None,
            "engagement_distance_ratio": None,
            "too_close_threshold_px": None,
            "distance_status": "unknown",
            "coaching_cue": "",
        }
        if len(tracks) < 2 or engagement_distance is None:
            return empty

        heights = [
            self._bbox_height(track.get("bbox"))
            for track in tracks[:2]
        ]
        heights = [height for height in heights if height is not None and height > 0]
        if not heights:
            return empty

        average_height = float(sum(heights) / len(heights))
        threshold = float(average_height * self.TOO_CLOSE_DISTANCE_RATIO)
        distance = float(engagement_distance)
        distance_ratio = float(distance / average_height)
        too_close = distance < threshold

        return {
            "average_fencer_height_px": average_height,
            "engagement_distance_ratio": distance_ratio,
            "too_close_threshold_px": threshold,
            "distance_status": "too_close" if too_close else "ok",
            "coaching_cue": (
                "Too close: recover distance."
                if too_close
                else "Distance OK."
            ),
        }

    def _bbox_from_skeleton(
        self,
        skeleton: Dict[str, Tuple[float, float]]
    ) -> Optional[List[float]]:
        points = [
            (float(coords[0]), float(coords[1]))
            for coords in skeleton.values()
            if self._valid_point(coords)
        ]
        if not points:
            return None

        x_coords = [point[0] for point in points]
        y_coords = [point[1] for point in points]
        return [
            min(x_coords),
            min(y_coords),
            max(x_coords),
            max(y_coords),
        ]

    def _center_from_bbox_or_skeleton(
        self,
        bbox: Optional[List[float]],
        skeleton: Dict[str, Tuple[float, float]]
    ) -> Optional[List[float]]:
        if bbox is not None:
            return [
                float((bbox[0] + bbox[2]) / 2.0),
                float((bbox[1] + bbox[3]) / 2.0),
            ]

        points = [
            (float(coords[0]), float(coords[1]))
            for coords in skeleton.values()
            if self._valid_point(coords)
        ]
        if not points:
            return None

        return [
            float(sum(point[0] for point in points) / len(points)),
            float(sum(point[1] for point in points) / len(points)),
        ]

    @staticmethod
    def _bbox_area(bbox: Optional[List[float]]) -> float:
        if bbox is None:
            return 0.0
        width = max(0.0, float(bbox[2]) - float(bbox[0]))
        height = max(0.0, float(bbox[3]) - float(bbox[1]))
        return float(width * height)

    @staticmethod
    def _bbox_height(bbox: Optional[List[float]]) -> Optional[float]:
        if bbox is None:
            return None
        return max(0.0, float(bbox[3]) - float(bbox[1]))

    @staticmethod
    def _as_float_list(
        values: Any,
        expected_len: int
    ) -> Optional[List[float]]:
        try:
            floats = [float(value) for value in values]
        except (TypeError, ValueError):
            return None
        if len(floats) != expected_len or not all(np.isfinite(floats)):
            return None
        return floats

    @staticmethod
    def _valid_point(coords: Any) -> bool:
        try:
            return (
                len(coords) == 2
                and np.isfinite(coords[0])
                and np.isfinite(coords[1])
            )
        except TypeError:
            return False

    @staticmethod
    def _json_skeleton(
        skeleton: Dict[str, Tuple[float, float]]
    ) -> Dict[str, List[float]]:
        return {
            joint_name: [float(coords[0]), float(coords[1])]
            for joint_name, coords in skeleton.items()
        }
