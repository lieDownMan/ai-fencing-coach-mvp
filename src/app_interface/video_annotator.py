"""Utilities for rendering tracked fencers onto processed video files."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

LEFT_COLOR = (255, 180, 0)
RIGHT_COLOR = (0, 180, 255)
OK_COLOR = (0, 180, 0)
WARN_COLOR = (0, 0, 255)
UNKNOWN_COLOR = (180, 180, 180)
TEXT_COLOR = (255, 255, 255)
PANEL_COLOR = (32, 32, 32)


def write_annotated_video(
    video_path: str,
    output_path: Path,
    tracking_frames: List[Dict[str, Any]],
    codec: str = "mp4v"
) -> Path:
    """Write a copy of the video with fencer boxes and distance warnings."""
    input_path = Path(video_path).expanduser()
    output_file = Path(output_path).expanduser()
    output_file.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video for annotation: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        cap.release()
        raise ValueError(f"Cannot read video dimensions: {input_path}")

    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(str(output_file), fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise OSError(f"Cannot open annotated video writer: {output_file}")

    frames_by_index = {
        int(frame.get("frame_index", frame_index)): frame
        for frame_index, frame in enumerate(tracking_frames or [])
    }

    frame_index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            tracking_frame = frames_by_index.get(frame_index)
            if tracking_frame is not None:
                frame = draw_tracking_overlay(frame, tracking_frame)
            writer.write(frame)
            frame_index += 1
    finally:
        cap.release()
        writer.release()

    return output_file


def draw_tracking_overlay(
    frame: np.ndarray,
    tracking_frame: Dict[str, Any]
) -> np.ndarray:
    """Draw fencer tracks, engagement distance, and too-close warnings."""
    if frame is None or frame.size == 0:
        raise ValueError("Frame must be a non-empty numpy array")

    output = frame.copy()
    tracks = tracking_frame.get("tracks", [])
    status = tracking_frame.get("distance_status", "unknown")
    distance_color = _distance_color(status)

    _draw_status_panel(output, tracking_frame, distance_color)
    _draw_tracks(output, tracks)
    _draw_distance_line(output, tracks, distance_color)
    return output


def _draw_status_panel(
    frame: np.ndarray,
    tracking_frame: Dict[str, Any],
    color: Tuple[int, int, int]
):
    frame_height, frame_width = frame.shape[:2]
    panel_height = min(82, max(54, frame_height // 8))
    cv2.rectangle(frame, (0, 0), (frame_width, panel_height), PANEL_COLOR, -1)
    cv2.rectangle(frame, (0, 0), (frame_width, panel_height), color, 3)

    status = tracking_frame.get("distance_status", "unknown")
    cue = tracking_frame.get("coaching_cue") or "Distance unavailable."
    distance = tracking_frame.get("engagement_distance_px")
    ratio = tracking_frame.get("engagement_distance_ratio")

    headline = "TOO CLOSE - recover distance" if status == "too_close" else cue
    cv2.putText(
        frame,
        headline,
        (12, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        color,
        2,
        cv2.LINE_AA,
    )

    detail = "Distance: unavailable"
    if distance is not None and ratio is not None:
        detail = f"Front-ankle distance: {float(distance):.1f}px ({float(ratio):.2f}x height)"
    elif distance is not None:
        detail = f"Front-ankle distance: {float(distance):.1f}px"
    cv2.putText(
        frame,
        detail,
        (12, 62),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        TEXT_COLOR,
        1,
        cv2.LINE_AA,
    )


def _draw_tracks(frame: np.ndarray, tracks: List[Dict[str, Any]]):
    for track in tracks:
        color = LEFT_COLOR if track.get("side") == "left" else RIGHT_COLOR
        bbox = _as_int_bbox(track.get("bbox"))
        center = _as_int_point(track.get("center"))
        label = track.get("track_id", "fencer")

        if bbox is not None:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                label,
                (x1, max(18, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                color,
                2,
                cv2.LINE_AA,
            )
        if center is not None:
            cv2.circle(frame, center, 5, color, -1)

        for point in (track.get("skeleton") or {}).values():
            joint = _as_int_point(point)
            if joint is not None:
                cv2.circle(frame, joint, 3, color, -1)


def _draw_distance_line(
    frame: np.ndarray,
    tracks: List[Dict[str, Any]],
    color: Tuple[int, int, int]
):
    if len(tracks) < 2:
        return

    left_point = _front_ankle_or_center(tracks[0])
    right_point = _front_ankle_or_center(tracks[1])
    if left_point is None or right_point is None:
        return

    cv2.line(frame, left_point, right_point, color, 3)
    midpoint = (
        int((left_point[0] + right_point[0]) / 2),
        int((left_point[1] + right_point[1]) / 2),
    )
    cv2.circle(frame, midpoint, 5, color, -1)


def _front_ankle_or_center(track: Dict[str, Any]) -> Optional[Tuple[int, int]]:
    skeleton = track.get("skeleton") or {}
    return _as_int_point(skeleton.get("front_ankle")) or _as_int_point(track.get("center"))


def _distance_color(status: str) -> Tuple[int, int, int]:
    if status == "too_close":
        return WARN_COLOR
    if status == "ok":
        return OK_COLOR
    return UNKNOWN_COLOR


def _as_int_bbox(value: Any) -> Optional[Tuple[int, int, int, int]]:
    if value is None:
        return None
    try:
        x1, y1, x2, y2 = value[:4]
    except (TypeError, ValueError):
        return None
    if not all(np.isfinite([x1, y1, x2, y2])):
        return None
    return int(x1), int(y1), int(x2), int(y2)


def _as_int_point(value: Any) -> Optional[Tuple[int, int]]:
    if value is None:
        return None
    try:
        x_coord, y_coord = value[:2]
    except (TypeError, ValueError):
        return None
    if not all(np.isfinite([x_coord, y_coord])):
        return None
    return int(x_coord), int(y_coord)
