from __future__ import annotations

import os
import shutil
import socket
import subprocess
import time
from html import escape
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import gradio as gr
import pandas as pd

from inference.heuristic_debug import (
    HEURISTIC_KEYS,
    build_debug_events,
    format_value,
    _target_detection,
)
from inference.sliding_window import FullVideoPipeline


_REPO_ROOT = Path(__file__).resolve().parent
_GRADIO_TEMP_DIR = _REPO_ROOT / "web_outputs" / "gradio_tmp"
_UPLOAD_DIR = _REPO_ROOT / "web_outputs" / "heuristic_debug" / "uploads"
_OUTPUT_DIR = _REPO_ROOT / "web_outputs" / "heuristic_debug" / "processed"
_LOG_DIR = _REPO_ROOT / "web_outputs" / "heuristic_debug" / "logs"
for _dir in (_GRADIO_TEMP_DIR, _UPLOAD_DIR, _OUTPUT_DIR, _LOG_DIR):
    _dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("GRADIO_TEMP_DIR", str(_GRADIO_TEMP_DIR))


PROCESSING_PROFILES = {
    "Balanced": {
        "max_pose_width": 960,
        "pose_every_n_frames": 1,
        "annotated_max_width": 1280,
    },
    "Fast": {
        "max_pose_width": 640,
        "pose_every_n_frames": 2,
        "annotated_max_width": 960,
    },
    "Full Quality": {
        "max_pose_width": None,
        "pose_every_n_frames": 1,
        "annotated_max_width": None,
    },
}

TERM_EXPLANATION = """### How to read this

- **System Alert**: the real runtime emitted this coaching warning for that action segment.
- **Metric Triggered**: the visualizer recomputed the raw heuristic metric and it crossed that heuristic threshold.
- **Primary Value**: the main number this heuristic compares against the threshold. The full supporting values are in `Metrics`.

If `Metric Triggered` is YES but `System Alert` is blank, the metric crossed its threshold in the debug pass, but the current runtime did not issue that warning for that segment. That can happen because the runtime only checks some heuristics for specific actions and training modes.
"""

HEURISTIC_NOTES = {
    "bounce_excessive": {
        "title": "bounce_excessive",
        "values": "pelvis vertical movement, full skeleton bbox height, ratio = pelvis_delta / bbox_height",
        "primary": "ratio",
        "threshold": "ratio > 0.10",
        "joints": "left_hip, right_hip, all detected joints for bbox height",
    },
    "lunge_overextension": {
        "title": "lunge_overextension",
        "values": "front ankle displacement, front knee angle at peak ankle displacement",
        "primary": "front_knee_angle",
        "threshold": "front_knee_angle < 90 deg",
        "joints": "front hip, front knee, front ankle",
    },
    "guard_dropped": {
        "title": "guard_dropped",
        "values": "whether front wrist is below pelvis, consecutive low-guard frames",
        "primary": "max_consecutive low-guard frames",
        "threshold": "> 10 frames outside Free Bouting, > 20 frames in Free Bouting",
        "joints": "front_wrist, left_hip, right_hip",
    },
    "foot_before_hand": {
        "title": "foot_before_hand",
        "values": "front wrist displacement peak frame, front ankle displacement peak frame",
        "primary": "ankle_peak_index - wrist_peak_index",
        "threshold": "ankle and wrist displacement > 5 px, ankle peak happens before wrist peak",
        "joints": "front_wrist, front_ankle",
    },
    "stance_too_high": {
        "title": "stance_too_high",
        "values": "front knee angle over the action window",
        "primary": "average front knee angle",
        "threshold": "avg_front_knee_angle > 160 deg",
        "joints": "front hip, front knee, front ankle",
    },
    "incomplete_arm_extension": {
        "title": "incomplete_arm_extension",
        "values": "front wrist displacement, arm angle at peak wrist extension",
        "primary": "arm_angle",
        "threshold": "arm_angle < 155 deg",
        "joints": "front_shoulder, front_elbow, front_wrist",
    },
    "pumping_the_arm": {
        "title": "pumping_the_arm",
        "values": "overall wrist travel direction, minimum early wrist retraction",
        "primary": "early_min_retract",
        "threshold": "early_wrist_retraction < -8 px",
        "joints": "front_wrist",
    },
    "over_parrying": {
        "title": "over_parrying",
        "values": "front wrist horizontal sweep, shoulder-width body reference, ratio",
        "primary": "wrist_sweep / shoulder_width",
        "threshold": "ratio > 2.0",
        "joints": "front_wrist, front_shoulder, opposite shoulder or pelvis fallback",
    },
    "wide_step": {
        "title": "wide_step",
        "values": "front/back ankle distance, shoulder-width proxy, step-width ratio",
        "primary": "max step_width / shoulder_width ratio",
        "threshold": "max_ratio > 3.0",
        "joints": "front ankle, back ankle, front_shoulder, pelvis",
    },
    "narrow_step": {
        "title": "narrow_step",
        "values": "front/back ankle distance, shoulder-width proxy, step-width ratio",
        "primary": "min step_width / shoulder_width ratio",
        "threshold": "min_ratio < 1.0",
        "joints": "front ankle, back ankle, front_shoulder, pelvis",
    },
    "center_of_mass_in_front": {
        "title": "center_of_mass_in_front",
        "values": "pelvis x-position between back ankle and front ankle",
        "primary": "max pelvis position ratio",
        "threshold": "max_ratio > 0.65",
        "joints": "front ankle, back ankle, left_hip, right_hip",
    },
    "center_of_mass_leaning_backward": {
        "title": "center_of_mass_leaning_backward",
        "values": "pelvis x-position between back ankle and front ankle",
        "primary": "min pelvis position ratio",
        "threshold": "min_ratio < 0.35",
        "joints": "front ankle, back ankle, left_hip, right_hip",
    },
}

DEBUG_CSS = """
#debug_help_box {
    font-size: 18px;
    line-height: 1.45;
}
#debug_summary_box {
    font-size: 17px;
    line-height: 1.45;
}
"""

CONNECTIONS = [
    ("nose", "front_shoulder"),
    ("front_shoulder", "front_elbow"),
    ("front_elbow", "front_wrist"),
    ("front_shoulder", "left_hip"),
    ("front_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
]


def _video_path(video_file: Any) -> Path:
    if isinstance(video_file, (str, Path)):
        return Path(video_file)
    if isinstance(video_file, dict):
        candidate = video_file.get("path") or video_file.get("name")
        if candidate:
            return Path(candidate)
    candidate = getattr(video_file, "path", None) or getattr(video_file, "name", None)
    if candidate:
        return Path(candidate)
    raise ValueError(f"Unsupported video input: {type(video_file).__name__}")


def _copy_video_to_workspace(video_file: Any) -> str:
    source = _video_path(video_file)
    if not source.exists():
        raise FileNotFoundError(f"Uploaded video file not found: {source}")

    suffix = source.suffix if source.suffix else ".mp4"
    safe_stem = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in source.stem)
    target = _UPLOAD_DIR / f"{safe_stem}_{int(time.time() * 1000)}{suffix}"
    shutil.copy2(source, target)
    return str(target)


def _validate_video_file(video_path: str) -> Dict[str, Any]:
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {path}")

    size_bytes = path.stat().st_size
    if size_bytes < 128:
        raise ValueError(
            f"Uploaded video is only {size_bytes} bytes. "
            "It looks incomplete or not fully saved yet."
        )

    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError(
                f"OpenCV could not open this video: {path.name}. "
                "The file may be corrupt, incomplete, or encoded in an unsupported format."
            )
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        ok, frame = cap.read()
        if not ok or frame is None:
            raise ValueError(
                f"OpenCV opened {path.name}, but could not read the first frame."
            )
        if width <= 0 or height <= 0:
            height, width = frame.shape[:2]
        if frame_count <= 0:
            frame_count = 1
        if fps <= 0:
            fps = 30.0
    finally:
        cap.release()

    return {
        "size_bytes": size_bytes,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "fps": fps,
    }


def _empty_table() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "Time",
            "Frames",
            "Action",
            "Heuristic",
            "System Alert (runtime)",
            "Metric Triggered (raw)",
            "Primary Value (main metric)",
            "Threshold",
            "Metrics",
            "Alert Time",
        ]
    )


def _events_to_dataframe(events: List[Dict[str, Any]]) -> pd.DataFrame:
    if not events:
        return _empty_table()

    rows = []
    for event in events:
        rows.append({
            "Time": event.get("time", ""),
            "Frames": f"{event.get('start_frame')} - {event.get('end_frame')}",
            "Action": event.get("action", ""),
            "Heuristic": event.get("heuristic", ""),
            "System Alert (runtime)": "YES" if event.get("alert") else "",
            "Metric Triggered (raw)": "YES" if event.get("metric_triggered") else "",
            "Primary Value (main metric)": format_value(event.get("primary_value")),
            "Threshold": event.get("threshold", ""),
            "Metrics": event.get("metrics", ""),
            "Alert Time": event.get("alert_time", ""),
        })
    return pd.DataFrame(rows, columns=_empty_table().columns)


def _event_status(event: Dict[str, Any]) -> str:
    if event.get("alert"):
        return "ALERT"
    if event.get("metric_triggered"):
        return "THRESHOLD"
    return "OK"


def _events_to_log_dataframe(events: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    metric_keys = sorted({
        key
        for event in events
        for key in (event.get("metric_values") or {}).keys()
    })
    for event in events:
        metric_values = event.get("metric_values") or {}
        row = {
            "time": event.get("time", ""),
            "start_frame": event.get("start_frame", ""),
            "end_frame": event.get("end_frame", ""),
            "action": event.get("action", ""),
            "heuristic": event.get("heuristic", ""),
            "status": _event_status(event),
            "system_alert": bool(event.get("alert")),
            "metric_triggered": bool(event.get("metric_triggered")),
            "primary_value": event.get("primary_value"),
            "threshold": event.get("threshold", ""),
            "alert_time": event.get("alert_time", ""),
        }
        for key in metric_keys:
            row[key] = metric_values.get(key, "")
        row["metrics"] = event.get("metrics", "")
        rows.append(row)

    columns = [
        "time",
        "start_frame",
        "end_frame",
        "action",
        "heuristic",
        "status",
        "system_alert",
        "metric_triggered",
        "primary_value",
        "threshold",
        "alert_time",
    ] + metric_keys + ["metrics"]
    return pd.DataFrame(rows, columns=columns)


def _write_debug_logs(events: List[Dict[str, Any]]) -> tuple[Optional[str], Optional[str], str]:
    if not events:
        return None, None, "<p>No heuristic rows to log.</p>"

    timestamp = int(time.time() * 1000)
    csv_path = _LOG_DIR / f"heuristic_debug_log_{timestamp}.csv"
    html_path = _LOG_DIR / f"heuristic_debug_log_{timestamp}.html"
    df = _events_to_log_dataframe(events)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    html = _build_log_html(df)
    html_path.write_text(html, encoding="utf-8")
    return str(csv_path), str(html_path), html


def _build_log_html(df: pd.DataFrame) -> str:
    headers = "".join(f"<th>{escape(str(col))}</th>" for col in df.columns)
    body_rows = []
    for _, row in df.iterrows():
        status = str(row.get("status", "OK"))
        row_class = "alert" if status == "ALERT" else "triggered" if status == "THRESHOLD" else "ok"
        row_style = _row_highlight_style(status)
        cells = []
        for col in df.columns:
            value = row.get(col, "")
            cell_style = row_style
            cell_class = ""
            if col == "primary_value" and status != "OK":
                cell_class = "primary"
                cell_style += "font-weight:700;color:#9a0000;"
            style_attr = f" style=\"{cell_style}\"" if cell_style else ""
            cells.append(
                f"<td class=\"{cell_class}\"{style_attr}>"
                f"{escape(_format_log_value(value))}</td>"
            )
        body_rows.append(f"<tr class=\"{row_class}\" style=\"{row_style}\">{''.join(cells)}</tr>")

    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<style>
body {{
    font-family: Arial, sans-serif;
    margin: 18px;
    color: #111;
}}
.legend {{
    font-size: 16px;
    margin-bottom: 14px;
}}
.table-wrap {{
    max-height: 78vh;
    overflow: auto;
    border: 1px solid #bbb;
}}
table {{
    border-collapse: collapse;
    width: 100%;
    font-size: 14px;
}}
th {{
    position: sticky;
    top: 0;
    background: #222;
    color: white;
    z-index: 1;
}}
th, td {{
    border: 1px solid #ccc;
    padding: 7px 9px;
    text-align: left;
    white-space: nowrap;
}}
tr.alert {{
    background: #ffd4d4;
}}
tr.triggered {{
    background: #fff0b8;
}}
tr.ok {{
    background: #fff;
}}
td.primary {{
    font-weight: 700;
    color: #9a0000;
}}
</style>
</head>
<body>
<h2>Heuristic Debug Log</h2>
<div class="legend">
<strong>Red</strong> = runtime System Alert. <strong>Yellow</strong> = raw Metric Triggered over threshold.
Primary value cells are bold when the row is over threshold.
</div>
<div class="table-wrap">
<table>
<thead><tr>{headers}</tr></thead>
<tbody>
{''.join(body_rows)}
</tbody>
</table>
</div>
</body>
</html>"""


def _row_highlight_style(status: str) -> str:
    if status == "ALERT":
        return "background-color:#ffd4d4;"
    if status == "THRESHOLD":
        return "background-color:#fff0b8;"
    return "background-color:#ffffff;"


def _format_log_value(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return format_value(value)


def _build_summary(events: List[Dict[str, Any]], video_info: Dict[str, Any]) -> str:
    if not events:
        return (
            "No action segments or target skeletons were available. "
            "Try a clearer clip or use Full Quality."
        )

    system_alerts = {}
    metric_triggers = {}
    for event in events:
        key = event.get("heuristic", "")
        if event.get("alert"):
            system_alerts[key] = system_alerts.get(key, 0) + 1
        if event.get("metric_triggered"):
            metric_triggers[key] = metric_triggers.get(key, 0) + 1

    alert_text = ", ".join(f"{key}: {count}" for key, count in sorted(system_alerts.items()))
    metric_text = ", ".join(f"{key}: {count}" for key, count in sorted(metric_triggers.items()))
    return "\n".join([
        f"Video: {video_info['width']}x{video_info['height']}, {video_info['frame_count']} frames, {video_info['fps']:.2f} fps",
        f"Rows: {len(events)}",
        f"System alerts: {alert_text or 'none'}",
        f"Metric triggers: {metric_text or 'none'}",
    ])


def _heuristic_note_text(heuristic_key: str) -> str:
    if heuristic_key == "all":
        lines = [
            TERM_EXPLANATION,
            "### Heuristic value map",
        ]
        for key in HEURISTIC_KEYS:
            note = HEURISTIC_NOTES[key]
            lines.append(
                f"- **{key}**: Primary Value = `{note['primary']}`; "
                f"threshold = `{note['threshold']}`; values = {note['values']}."
            )
        return "\n\n".join(lines)

    note = HEURISTIC_NOTES.get(heuristic_key)
    if not note:
        return TERM_EXPLANATION
    return (
        f"{TERM_EXPLANATION}\n\n"
        f"### {note['title']}\n\n"
        f"- **Skeleton values used**: {note['values']}\n"
        f"- **Joints used**: {note['joints']}\n"
        f"- **Primary Value**: `{note['primary']}`\n"
        f"- **Threshold / parameter**: `{note['threshold']}`"
    )


def _overlay_note_lines(selected_heuristic: str) -> List[str]:
    base = [
        "System Alert = runtime warning emitted",
        "Metric Triggered = raw metric crossed threshold",
        "Primary Value = main number compared with threshold",
    ]
    if selected_heuristic == "all":
        return base + ["Select one heuristic for value/parameter notes"]

    note = HEURISTIC_NOTES.get(selected_heuristic)
    if not note:
        return base
    return base + [
        f"Values: {note['values']}",
        f"Primary: {note['primary']}",
        f"Threshold: {note['threshold']}",
    ]


def analyze_heuristics(
    video_file: Any,
    target_side: str,
    training_mode: str,
    heuristic_key: str,
    processing_profile: str,
    progress: gr.Progress = gr.Progress(),
):
    if not video_file:
        return None, _empty_table(), "Upload a video first.", "", None, None

    profile = PROCESSING_PROFILES.get(processing_profile, PROCESSING_PROFILES["Balanced"])
    try:
        progress(0.02, desc="Copying upload")
        input_video_path = _copy_video_to_workspace(video_file)
        video_info = _validate_video_file(input_video_path)
    except (FileNotFoundError, ValueError) as exc:
        progress(1.0, desc="Stopped")
        return None, _empty_table(), f"Video upload problem: {exc}", "", None, None

    pipeline = FullVideoPipeline(
        target_side=target_side,
        training_mode=training_mode,
        max_pose_width=profile["max_pose_width"],
        pose_every_n_frames=profile["pose_every_n_frames"],
    )
    results = pipeline.process_video(
        input_video_path,
        progress_callback=lambda fraction, desc: progress(fraction, desc=desc),
    )

    progress(0.90, desc="Computing heuristic metrics")
    events = build_debug_events(
        results,
        heuristic_key=heuristic_key,
        target_side=target_side,
        training_mode=training_mode,
        fps=video_info["fps"],
    )

    progress(0.94, desc="Rendering debug overlay")
    output_path = str(_OUTPUT_DIR / f"heuristic_debug_{int(time.time() * 1000)}.mp4")
    render_debug_video(
        input_video_path,
        output_path,
        results,
        events,
        selected_heuristic=heuristic_key,
        fps=video_info["fps"],
        max_width=profile["annotated_max_width"],
    )

    progress(0.98, desc="Optimizing video")
    _try_optimize_mp4(output_path)

    progress(0.99, desc="Writing debug logs")
    csv_log, html_log, html_preview = _write_debug_logs(events)

    progress(1.0, desc="Done")
    return (
        output_path,
        _events_to_dataframe(events),
        _build_summary(events, video_info),
        html_preview,
        csv_log,
        html_log,
    )


def render_debug_video(
    input_path: str,
    output_path: str,
    report: Dict[str, Any],
    events: List[Dict[str, Any]],
    selected_heuristic: str,
    fps: float,
    max_width: Optional[int] = None,
) -> str:
    tracking = report.get("two_fencer_tracking", {})
    frames_dict = {
        frame.get("frame_index"): frame
        for frame in tracking.get("frames", [])
        if frame.get("frame_index") is not None
    }
    locked_track_id = tracking.get("locked_track_id")

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {input_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if width <= 0 or height <= 0:
        cap.release()
        raise ValueError(f"Cannot read video dimensions: {input_path}")

    output_width, output_height = _output_size(width, height, max_width)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps or 30.0, (output_width, output_height))
    if not out.isOpened():
        cap.release()
        raise ValueError(f"Cannot open video writer: {output_path}")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_info = frames_dict.get(frame_idx)
        current_events = _events_for_frame(events, frame_idx, selected_heuristic)
        has_alert = any(event.get("alert") for event in current_events)
        has_metric_trigger = any(event.get("metric_triggered") for event in current_events)

        if frame_info:
            target = _target_detection(frame_info.get("tracks", []), locked_track_id)
            if target:
                _draw_target(frame, target, has_alert, has_metric_trigger)

        _draw_hud(frame, frame_idx, fps, current_events, selected_heuristic)
        _draw_notes_panel(frame, selected_heuristic)

        if (output_width, output_height) != (width, height):
            frame = cv2.resize(frame, (output_width, output_height), interpolation=cv2.INTER_AREA)
        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    return output_path


def _output_size(width: int, height: int, max_width: Optional[int]) -> tuple[int, int]:
    if max_width and width > max_width:
        output_width = int(max_width)
        output_height = int(round(height * (output_width / width)))
    else:
        output_width = width
        output_height = height
    output_width = max(2, output_width - (output_width % 2))
    output_height = max(2, output_height - (output_height % 2))
    return output_width, output_height


def _events_for_frame(
    events: List[Dict[str, Any]],
    frame_idx: int,
    selected_heuristic: str,
) -> List[Dict[str, Any]]:
    candidates = [
        event
        for event in events
        if int(event.get("start_frame", -1)) <= frame_idx <= int(event.get("end_frame", -1))
    ]
    if selected_heuristic != "all":
        candidates = [event for event in candidates if event.get("heuristic") == selected_heuristic]
    candidates.sort(
        key=lambda event: (
            not bool(event.get("alert")),
            not bool(event.get("metric_triggered")),
            str(event.get("heuristic", "")),
        )
    )
    return candidates[:5]


def _draw_target(
    frame,
    target: Dict[str, Any],
    has_alert: bool,
    has_metric_trigger: bool,
) -> None:
    color = (0, 0, 255) if has_alert else (0, 180, 255) if has_metric_trigger else (0, 255, 0)
    bbox = target.get("bbox")
    if bbox and len(bbox) == 4:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

    skeleton = target.get("skeleton") or {}
    for pt in skeleton.values():
        cv2.circle(frame, (int(pt[0]), int(pt[1])), 4, (0, 255, 255), -1)
    for joint1, joint2 in CONNECTIONS:
        if joint1 in skeleton and joint2 in skeleton:
            pt1 = (int(skeleton[joint1][0]), int(skeleton[joint1][1]))
            pt2 = (int(skeleton[joint2][0]), int(skeleton[joint2][1]))
            cv2.line(frame, pt1, pt2, (0, 255, 255), 2)


def _draw_hud(
    frame,
    frame_idx: int,
    fps: float,
    current_events: List[Dict[str, Any]],
    selected_heuristic: str,
) -> None:
    _, width = frame.shape[:2]
    panel_width = min(width - 20, 1380)
    lines = [
        f"t={frame_idx / (fps or 30.0):.2f}s frame={frame_idx} mode={selected_heuristic}"
    ]
    if not current_events:
        lines.append("No action segment at this frame")
    else:
        for event in current_events:
            state = "ALERT" if event.get("alert") else "METRIC" if event.get("metric_triggered") else "OK"
            lines.append(
                f"{state} {event.get('heuristic')}: "
                f"value={format_value(event.get('primary_value'))} | {event.get('metrics', '')}"
            )
            if event.get("alert_time"):
                lines.append(f"alert_time={event.get('alert_time')} action={event.get('action')}")

    wrapped_lines: List[str] = []
    for line in lines:
        wrapped_lines.extend(_wrap_text(line, max_chars=108))
    lines = wrapped_lines[:10]
    line_gap = 34
    panel_height = 32 + line_gap * len(lines)
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (10 + panel_width, 10 + panel_height), (25, 25, 25), -1)
    cv2.addWeighted(overlay, 0.76, frame, 0.24, 0, frame)

    border_color = (180, 180, 180)
    if any(event.get("alert") for event in current_events):
        border_color = (0, 0, 255)
    elif any(event.get("metric_triggered") for event in current_events):
        border_color = (0, 180, 255)
    cv2.rectangle(frame, (10, 10), (10 + panel_width, 10 + panel_height), border_color, 4)

    y = 48
    for line in lines:
        color = (255, 255, 255)
        if line.startswith("ALERT"):
            color = (0, 0, 255)
        elif line.startswith("METRIC"):
            color = (0, 220, 255)
        cv2.putText(frame, line, (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.82, color, 2)
        y += line_gap


def _draw_notes_panel(frame, selected_heuristic: str) -> None:
    height, width = frame.shape[:2]
    lines: List[str] = []
    for line in _overlay_note_lines(selected_heuristic):
        lines.extend(_wrap_text(line, max_chars=92))
    lines = lines[:8]
    if not lines:
        return

    panel_width = min(width - 20, 1180)
    line_gap = 29
    panel_height = 28 + line_gap * len(lines)
    x1 = 10
    y1 = max(10, height - panel_height - 12)
    x2 = x1 + panel_width
    y2 = y1 + panel_height

    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (35, 35, 35), -1)
    cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (120, 120, 120), 2)

    y = y1 + 33
    for i, line in enumerate(lines):
        color = (230, 230, 230) if i < 3 else (210, 240, 255)
        cv2.putText(frame, line, (x1 + 18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.70, color, 2)
        y += line_gap


def _wrap_text(text: str, max_chars: int) -> List[str]:
    if len(text) <= max_chars:
        return [text]
    words = text.split(" ")
    lines = []
    current = ""
    for word in words:
        if not current:
            current = word
        elif len(current) + 1 + len(word) <= max_chars:
            current += " " + word
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def _try_optimize_mp4(output_path: str) -> None:
    tmp_video = output_path.replace(".mp4", "_web.mp4")
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                output_path,
                "-vcodec",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                tmp_video,
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        os.replace(tmp_video, output_path)
    except (FileNotFoundError, subprocess.CalledProcessError):
        if os.path.exists(tmp_video):
            os.remove(tmp_video)


def _pick_gradio_port(default_port: int) -> int:
    env_port = os.getenv("GRADIO_SERVER_PORT")
    if env_port:
        return int(env_port)

    for port in range(default_port, default_port + 20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.2)
            if sock.connect_ex(("127.0.0.1", port)) != 0:
                return port
    return default_port


with gr.Blocks(title="Fencing Heuristic Visualizer") as app:
    gr.Markdown("# Fencing Heuristic Visualizer")
    with gr.Row():
        with gr.Column(scale=1):
            video_input = gr.File(
                label="Upload Video",
                file_types=[".mp4", ".mov", ".avi", ".mkv"],
                type="filepath",
            )
            target_side = gr.Radio(["left", "right"], value="left", label="Target Fencer")
            training_mode = gr.Radio(
                ["Footwork", "Target Practice", "Free Bouting"],
                value="Footwork",
                label="Training Mode",
            )
            heuristic_key = gr.Dropdown(
                ["all"] + HEURISTIC_KEYS,
                value="all",
                label="Heuristic Mode",
            )
            processing_profile = gr.Radio(
                list(PROCESSING_PROFILES.keys()),
                value="Balanced",
                label="Processing",
            )
            heuristic_help = gr.Markdown(
                value=_heuristic_note_text("all"),
                elem_id="debug_help_box",
            )
            analyze_btn = gr.Button("Run Heuristic Debug", variant="primary")
        with gr.Column(scale=2):
            video_output = gr.Video(label="Debug Overlay")
            summary_output = gr.Markdown(
                value="Upload a video to inspect heuristic metrics.",
                elem_id="debug_summary_box",
            )
            event_table = gr.Dataframe(label="Heuristic Events", wrap=True)
            with gr.Accordion("Downloadable Debug Log", open=True):
                with gr.Row():
                    csv_log_output = gr.File(label="CSV Log")
                    html_log_output = gr.File(label="Highlighted HTML Log")
                log_preview = gr.HTML(label="Highlighted Log Preview")

    heuristic_key.change(_heuristic_note_text, [heuristic_key], [heuristic_help])

    analyze_btn.click(
        analyze_heuristics,
        [
            video_input,
            target_side,
            training_mode,
            heuristic_key,
            processing_profile,
        ],
        [
            video_output,
            event_table,
            summary_output,
            log_preview,
            csv_log_output,
            html_log_output,
        ],
    )


if __name__ == "__main__":
    port = _pick_gradio_port(7862)
    print(f"Launching Fencing Heuristic Visualizer at http://127.0.0.1:{port}")
    app.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=os.getenv("GRADIO_SHARE", "0") == "1",
        allowed_paths=[str(_OUTPUT_DIR), str(_UPLOAD_DIR), str(_LOG_DIR), str(_GRADIO_TEMP_DIR)],
        css=DEBUG_CSS,
    )
