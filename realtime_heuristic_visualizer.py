from __future__ import annotations

import argparse
import csv
import os
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, List

import cv2

from inference.heuristic_debug import HEURISTIC_KEYS, compute_heuristic_metric, format_metrics, format_value
from inference.target_tracker import TargetTracker
from src.pose_estimation import PoseEstimator


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


def _parse_source(source: str):
    try:
        return int(source)
    except ValueError:
        return source


def _open_capture(source):
    if isinstance(source, int) and os.name == "nt":
        return cv2.VideoCapture(source, cv2.CAP_DSHOW)
    return cv2.VideoCapture(source)


def _draw_skeleton(frame, skeleton: Dict[str, Any], color=(0, 255, 255)) -> None:
    if not skeleton:
        return
    for point in skeleton.values():
        cv2.circle(frame, (int(point[0]), int(point[1])), 4, color, -1)
    for joint_a, joint_b in CONNECTIONS:
        if joint_a in skeleton and joint_b in skeleton:
            pt_a = (int(skeleton[joint_a][0]), int(skeleton[joint_a][1]))
            pt_b = (int(skeleton[joint_b][0]), int(skeleton[joint_b][1]))
            cv2.line(frame, pt_a, pt_b, color, 2)


def _draw_bbox(frame, bbox, color) -> None:
    if not bbox or len(bbox) != 4:
        return
    x1, y1, x2, y2 = [int(v) for v in bbox]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)


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


def _draw_hud(
    frame,
    frame_idx: int,
    target_side: str,
    training_mode: str,
    heuristic_key: str,
    locked_track_id,
    buffer_len: int,
    window_size: int,
    metrics: List[Dict[str, Any]],
) -> None:
    height, width = frame.shape[:2]
    triggered = [row for row in metrics if row["triggered"]]
    ok_rows = [row for row in metrics if not row["triggered"]]
    shown = (triggered + ok_rows)[:8]

    lines = [
        f"frame={frame_idx} target={target_side} mode={training_mode} heuristic={heuristic_key}",
        f"locked_track_id={locked_track_id if locked_track_id is not None else 'position'} buffer={buffer_len}/{window_size}",
        "q quit | r reset target lock",
    ]
    if not shown:
        lines.append("No target skeleton yet")
    else:
        for row in shown:
            state = "TRIGGER" if row["triggered"] else "OK"
            lines.append(
                f"{state} {row['heuristic']}: value={format_value(row['primary_value'])} | {row['metrics']}"
            )

    wrapped = []
    for line in lines:
        wrapped.extend(_wrap_text(line, 106))
    wrapped = wrapped[:12]

    panel_width = min(width - 20, 1320)
    line_gap = 31
    panel_height = 30 + line_gap * len(wrapped)
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (10 + panel_width, 10 + panel_height), (25, 25, 25), -1)
    cv2.addWeighted(overlay, 0.76, frame, 0.24, 0, frame)
    border = (0, 0, 255) if triggered else (120, 120, 120)
    cv2.rectangle(frame, (10, 10), (10 + panel_width, 10 + panel_height), border, 3)

    y = 42
    for line in wrapped:
        color = (255, 255, 255)
        if line.startswith("TRIGGER"):
            color = (0, 0, 255)
        elif line.startswith("OK"):
            color = (0, 220, 255)
        cv2.putText(frame, line, (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.72, color, 2)
        y += line_gap

    if triggered:
        cv2.putText(
            frame,
            "THRESHOLD TRIGGERED",
            (24, max(50, height - 30)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            3,
        )


def _compute_metrics(
    skeleton_window: List[Dict[str, Any]],
    heuristic_key: str,
    target_side: str,
    training_mode: str,
) -> List[Dict[str, Any]]:
    if not skeleton_window:
        return []
    keys = HEURISTIC_KEYS if heuristic_key == "all" else [heuristic_key]
    rows = []
    for key in keys:
        metric = compute_heuristic_metric(
            key,
            skeleton_window,
            target_side=target_side,
            training_mode=training_mode,
        )
        rows.append({
            "heuristic": key,
            "triggered": metric.triggered,
            "primary_value": metric.primary_value,
            "threshold": metric.threshold,
            "metric_values": metric.values,
            "metrics": format_metrics(metric),
        })
    return rows


def _append_log_rows(
    log_rows: List[Dict[str, Any]],
    frame_idx: int,
    metrics: List[Dict[str, Any]],
    buffer_len: int,
    window_size: int,
) -> None:
    for row in metrics:
        metric_values = row.get("metric_values") or {}
        item = {
            "frame": frame_idx,
            "heuristic": row["heuristic"],
            "status": "TRIGGER" if row["triggered"] else "OK",
            "primary_value": row["primary_value"],
            "threshold": row["threshold"],
            "sample_count": buffer_len,
            "window_size": window_size,
            "metrics": row["metrics"],
        }
        item.update(metric_values)
        log_rows.append(item)


def _write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Realtime frame-by-frame heuristic visualizer.")
    parser.add_argument("--source", default="0", help="Camera source index or video path")
    parser.add_argument("--target-side", default="left", choices=["left", "right"])
    parser.add_argument("--mode", default="Footwork", choices=["Footwork", "Target Practice", "Free Bouting"])
    parser.add_argument("--heuristic", default="all", choices=["all"] + HEURISTIC_KEYS)
    parser.add_argument("--window-size", type=int, default=28, help="Rolling skeleton window size")
    parser.add_argument("--pose-model", default=None, help="Path/name for YOLO pose weights")
    parser.add_argument("--pose-backend", default="ultralytics", choices=["ultralytics", "mock"])
    parser.add_argument("--log-csv", default=None, help="Optional CSV path for per-frame heuristic metrics")
    args = parser.parse_args()

    source = _parse_source(args.source)
    cap = _open_capture(source)
    if not cap.isOpened():
        print(f"Error: Cannot open video source {source}")
        return 1

    pose_estimator = PoseEstimator(model_path=args.pose_model, backend=args.pose_backend)
    tracker = TargetTracker(target_side=args.target_side)
    skeleton_window = deque(maxlen=max(1, args.window_size))
    log_rows: List[Dict[str, Any]] = []
    frame_idx = 0

    print("==============================================")
    print(" Realtime Heuristic Visualizer")
    print(f" Source: {source} | Target: {args.target_side} | Mode: {args.mode}")
    print(f" Heuristic: {args.heuristic} | Window: {args.window_size} frames")
    print(" Press q to quit, r to reset target lock.")
    print("==============================================")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] End of stream or camera frame unavailable.")
            break

        detections = pose_estimator.extract_frame_fencers(frame, persist_track=True)
        target_skel, opp_skel = tracker.process_frame_detections(detections, frame_idx)

        if target_skel:
            skeleton_window.append(target_skel)
        else:
            skeleton_window.append({})

        clean_window = [skel for skel in skeleton_window if skel]
        metrics = _compute_metrics(clean_window, args.heuristic, args.target_side, args.mode)
        any_triggered = any(row["triggered"] for row in metrics)

        if target_skel:
            _draw_skeleton(frame, target_skel, color=(0, 255, 255))
            _draw_bbox(frame, tracker.last_known_bbox, (0, 0, 255) if any_triggered else (0, 255, 0))
        if opp_skel:
            _draw_skeleton(frame, opp_skel, color=(160, 160, 160))

        _draw_hud(
            frame,
            frame_idx=frame_idx,
            target_side=args.target_side,
            training_mode=args.mode,
            heuristic_key=args.heuristic,
            locked_track_id=tracker.locked_track_id,
            buffer_len=len(clean_window),
            window_size=args.window_size,
            metrics=metrics,
        )

        if args.log_csv:
            _append_log_rows(log_rows, frame_idx, metrics, len(clean_window), args.window_size)

        cv2.imshow("Realtime Heuristic Visualizer", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("r"):
            tracker.reset()
            skeleton_window.clear()
            print("[INFO] Target lock reset.")

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()
    if args.log_csv:
        _write_csv(args.log_csv, log_rows)
        print(f"[INFO] Wrote realtime heuristic log: {args.log_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
