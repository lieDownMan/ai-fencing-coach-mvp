"""Utilities for rendering tracked fencers onto processed video files."""

import logging
from pathlib import Path
import shutil
import subprocess
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.tracking import PatternAnalyzer

logger = logging.getLogger(__name__)

LEFT_COLOR = (255, 180, 0)
RIGHT_COLOR = (0, 180, 255)
OK_COLOR = (0, 180, 0)
WARN_COLOR = (0, 0, 255)
UNKNOWN_COLOR = (180, 180, 180)
TEXT_COLOR = (255, 255, 255)
PANEL_COLOR = (32, 32, 32)


def _scaled_dimensions(
    width: int,
    height: int,
    max_width: Optional[int] = None
) -> Tuple[int, int]:
    """Return output dimensions, preserving aspect ratio when downscaling."""
    if max_width is None or max_width <= 0 or width <= max_width:
        return int(width), int(height)

    scaled_height = int(round(height * (max_width / width)))
    # Keep dimensions even for broad MP4 player/browser compatibility.
    scaled_width = max(2, int(max_width) - int(max_width) % 2)
    scaled_height = max(2, scaled_height - scaled_height % 2)
    return scaled_width, scaled_height


def write_annotated_video(
    video_path: str,
    output_path: Path,
    tracking_frames: List[Dict[str, Any]],
    codec: str = "mp4v",
    classifications: Optional[List[Tuple[int, float]]] = None,
    window_size: int = 28,
    window_stride: int = 14,
    fencer_heights_cm: Optional[Dict[str, float]] = None,
    max_width: Optional[int] = None
) -> Path:
    """Write a copy of the video with fencer boxes and dual HUD panels."""
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

    output_width, output_height = _scaled_dimensions(
        width=width,
        height=height,
        max_width=max_width,
    )

    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(
        str(output_file),
        fourcc,
        fps,
        (output_width, output_height),
    )
    if not writer.isOpened():
        cap.release()
        raise OSError(f"Cannot open annotated video writer: {output_file}")

    frames_by_index = {
        int(frame.get("frame_index", frame_index)): frame
        for frame_index, frame in enumerate(tracking_frames or [])
    }
    actions_by_frame = _build_action_lookup(
        classifications or [],
        window_size=window_size,
        window_stride=window_stride,
    )
    motion_by_frame = _build_motion_lookup(tracking_frames or [], fps=fps)

    frame_index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            tracking_frame = frames_by_index.get(frame_index)
            if tracking_frame is not None:
                frame = draw_tracking_overlay(
                    frame,
                    tracking_frame,
                    fencer_heights_cm=fencer_heights_cm,
                    global_action=actions_by_frame.get(frame_index),
                    track_motion=motion_by_frame.get(frame_index, {}),
                )
            if (output_width, output_height) != (width, height):
                frame = cv2.resize(frame, (output_width, output_height))
            writer.write(frame)
            frame_index += 1
    finally:
        cap.release()
        writer.release()

    if max_width is not None:
        return _transcode_mp4_for_browser(output_file)
    return output_file


def _transcode_mp4_for_browser(
    video_path: Path,
    ffmpeg_path: Optional[str] = None
) -> Path:
    """Transcode an MP4 to H.264/yuv420p for browser playback when possible."""
    ffmpeg = ffmpeg_path or shutil.which("ffmpeg")
    if not ffmpeg:
        logger.warning("ffmpeg not found; leaving annotated video in OpenCV codec")
        return video_path

    output_path = Path(video_path)
    temp_path = output_path.with_name(f"{output_path.stem}_browser{output_path.suffix}")
    command = [
        ffmpeg,
        "-y",
        "-i",
        str(output_path),
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-an",
        str(temp_path),
    ]

    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        logger.warning("Could not run ffmpeg for browser MP4 transcode: %s", exc)
        return output_path

    if result.returncode != 0:
        logger.warning(
            "ffmpeg browser MP4 transcode failed: %s",
            result.stderr.strip() if result.stderr else "unknown error",
        )
        if temp_path.exists():
            temp_path.unlink()
        return output_path

    temp_path.replace(output_path)
    return output_path


def draw_tracking_overlay(
    frame: np.ndarray,
    tracking_frame: Dict[str, Any],
    fencer_heights_cm: Optional[Dict[str, float]] = None,
    global_action: Optional[Dict[str, Any]] = None,
    track_motion: Optional[Dict[str, Dict[str, Any]]] = None
) -> np.ndarray:
    """Draw fencer tracks, engagement distance, and per-fencer HUDs."""
    if frame is None or frame.size == 0:
        raise ValueError("Frame must be a non-empty numpy array")

    output = frame.copy()
    tracks = tracking_frame.get("tracks", [])
    status = tracking_frame.get("distance_status", "unknown")
    distance_color = _distance_color(status)

    _draw_status_panel(output, tracking_frame, distance_color)
    _draw_tracks(output, tracks)
    _draw_distance_line(output, tracks, distance_color)
    _draw_fencer_huds(
        output,
        tracks,
        tracking_frame,
        fencer_heights_cm=fencer_heights_cm or {},
        global_action=global_action,
        track_motion=track_motion or {},
    )
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
        detail = f"Front-ankle distance: {float(distance):.1f}px ({float(ratio):.2f}x avg height)"
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


def _draw_fencer_huds(
    frame: np.ndarray,
    tracks: List[Dict[str, Any]],
    tracking_frame: Dict[str, Any],
    fencer_heights_cm: Dict[str, float],
    global_action: Optional[Dict[str, Any]],
    track_motion: Dict[str, Dict[str, Any]],
):
    if not tracks:
        return

    frame_height, frame_width = frame.shape[:2]
    panel_width = max(150, min(360, (frame_width - 36) // 2))
    panel_height = min(142, max(112, frame_height // 6))
    panel_top = min(
        max(88, frame_height // 9),
        max(8, frame_height - panel_height - 8),
    )
    panels = {
        "left": (8, panel_top),
        "right": (frame_width - panel_width - 8, panel_top),
    }

    for track in tracks:
        side = str(track.get("side", "unknown"))
        if side not in {"left", "right"}:
            continue
        color = LEFT_COLOR if side == "left" else RIGHT_COLOR
        title = str(track.get("track_id") or f"fencer_{side}")
        lines = _fencer_hud_lines(
            track,
            tracking_frame,
            fencer_heights_cm=fencer_heights_cm,
            global_action=global_action,
            motion=track_motion.get(title, {}),
        )
        _draw_hud_panel(
            frame,
            top_left=panels[side],
            size=(panel_width, panel_height),
            color=color,
            title=title,
            lines=lines,
        )


def _fencer_hud_lines(
    track: Dict[str, Any],
    tracking_frame: Dict[str, Any],
    fencer_heights_cm: Dict[str, float],
    global_action: Optional[Dict[str, Any]],
    motion: Dict[str, Any],
) -> List[str]:
    height_px = _bbox_height(track.get("bbox"))
    distance_px = _as_float_or_none(tracking_frame.get("engagement_distance_px"))
    height_cm = _height_for_track(track, fencer_heights_cm)

    height_line = f"Height: {height_cm:.0f} cm" if height_cm is not None else "Height: auto"

    distance_line = "Distance: unavailable"
    status_line = "Status: unknown"
    if distance_px is not None and height_px is not None and height_px > 0:
        ratio = distance_px / height_px
        status = "TOO CLOSE" if ratio < 1.0 else "OK"
        distance_line = f"Distance: {ratio:.2f}x height"
        if height_cm is not None:
            distance_cm = ratio * height_cm
            distance_line += f" (~{distance_cm:.0f} cm)"
        status_line = f"Status: {status}"

    speed = _as_float_or_none(motion.get("speed_height_per_s"))
    movement = motion.get("movement") or "unknown"
    speed_line = "Speed: unknown"
    if speed is not None:
        speed_line = f"Speed: {speed:.2f} height/s"
    movement_line = f"Movement: {movement}"

    action_line = "Global action: unknown"
    if global_action:
        action = global_action.get("action", "unknown")
        confidence = _as_float_or_none(global_action.get("confidence"))
        if confidence is not None:
            action_line = f"Global action: {action} ({confidence:.2f})"
        else:
            action_line = f"Global action: {action}"

    return [
        height_line,
        distance_line,
        status_line,
        speed_line,
        movement_line,
        action_line,
    ]


def _draw_hud_panel(
    frame: np.ndarray,
    top_left: Tuple[int, int],
    size: Tuple[int, int],
    color: Tuple[int, int, int],
    title: str,
    lines: List[str],
):
    x_coord, y_coord = top_left
    width, height = size
    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (x_coord, y_coord),
        (x_coord + width, y_coord + height),
        PANEL_COLOR,
        -1,
    )
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)
    cv2.rectangle(
        frame,
        (x_coord, y_coord),
        (x_coord + width, y_coord + height),
        color,
        2,
    )
    cv2.putText(
        frame,
        title,
        (x_coord + 10, y_coord + 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        color,
        2,
        cv2.LINE_AA,
    )

    for line_index, line in enumerate(lines[:6]):
        cv2.putText(
            frame,
            line,
            (x_coord + 10, y_coord + 48 + line_index * 17),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            TEXT_COLOR,
            1,
            cv2.LINE_AA,
        )


def _build_action_lookup(
    classifications: List[Tuple[int, float]],
    window_size: int,
    window_stride: int,
) -> Dict[int, Dict[str, Any]]:
    action_by_frame: Dict[int, Dict[str, Any]] = {}
    window_size = max(1, int(window_size or 1))
    window_stride = max(1, int(window_stride or 1))

    for window_index, classification in enumerate(classifications):
        try:
            class_idx, confidence = classification
        except (TypeError, ValueError):
            continue
        class_idx = int(class_idx)
        action = PatternAnalyzer.ACTION_CLASSES.get(class_idx, "unknown")
        action_payload = {
            "action": action,
            "confidence": float(confidence),
            "window_index": window_index,
        }
        start_frame = window_index * window_stride
        for frame_index in range(start_frame, start_frame + window_size):
            action_by_frame[frame_index] = action_payload

    return action_by_frame


def _build_motion_lookup(
    tracking_frames: List[Dict[str, Any]],
    fps: float,
) -> Dict[int, Dict[str, Dict[str, Any]]]:
    motion_by_frame: Dict[int, Dict[str, Dict[str, Any]]] = {}
    previous_by_track: Dict[str, Tuple[int, Tuple[float, float], float, str]] = {}
    fps = float(fps or 30.0)

    for fallback_index, tracking_frame in enumerate(tracking_frames):
        frame_index = int(tracking_frame.get("frame_index", fallback_index))
        for track in tracking_frame.get("tracks", []):
            track_id = str(track.get("track_id") or "")
            center = _as_float_point(track.get("center"))
            height_px = _bbox_height(track.get("bbox"))
            side = str(track.get("side") or "unknown")
            if not track_id or center is None or height_px is None or height_px <= 0:
                continue

            previous = previous_by_track.get(track_id)
            if previous is not None:
                previous_frame, previous_center, previous_height, _ = previous
                frame_delta = max(1, frame_index - previous_frame)
                dt_seconds = frame_delta / fps
                dx = center[0] - previous_center[0]
                dy = center[1] - previous_center[1]
                height_reference = max(height_px, previous_height, 1.0)
                speed_px_per_s = float(np.hypot(dx, dy) / dt_seconds)
                speed_height_per_s = speed_px_per_s / height_reference
                x_speed_height_per_s = (dx / dt_seconds) / height_reference
                motion_by_frame.setdefault(frame_index, {})[track_id] = {
                    "speed_height_per_s": speed_height_per_s,
                    "movement": _movement_label(side, x_speed_height_per_s),
                }

            previous_by_track[track_id] = (
                frame_index,
                center,
                float(height_px),
                side,
            )

    return motion_by_frame


def _movement_label(side: str, x_speed_height_per_s: float) -> str:
    if abs(x_speed_height_per_s) < 0.10:
        return "holding"
    if (side == "left" and x_speed_height_per_s > 0) or (
        side == "right" and x_speed_height_per_s < 0
    ):
        return "advancing"
    return "retreating"


def _height_for_track(
    track: Dict[str, Any],
    fencer_heights_cm: Dict[str, float],
) -> Optional[float]:
    track_id = str(track.get("track_id") or "")
    side = str(track.get("side") or "")
    candidates = [
        track_id,
        side,
        f"fencer_{side[0].upper()}" if side in {"left", "right"} else "",
    ]
    for key in candidates:
        if key and key in fencer_heights_cm:
            value = _as_float_or_none(fencer_heights_cm[key])
            if value is not None and value > 0:
                return value
    return None


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


def _bbox_height(value: Any) -> Optional[float]:
    bbox = _as_int_bbox(value)
    if bbox is None:
        return None
    return float(max(0, bbox[3] - bbox[1]))


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


def _as_float_point(value: Any) -> Optional[Tuple[float, float]]:
    if value is None:
        return None
    try:
        x_coord, y_coord = value[:2]
    except (TypeError, ValueError):
        return None
    if not all(np.isfinite([x_coord, y_coord])):
        return None
    return float(x_coord), float(y_coord)


def _as_float_or_none(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(result):
        return None
    return result
