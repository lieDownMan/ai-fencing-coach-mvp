from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from .heuristics_engine import FRONT_LIMBS, calc_angle, _get_joint, _pelvis_center


HEURISTIC_KEYS = [
    "bounce_excessive",
    "lunge_overextension",
    "guard_dropped",
    "foot_before_hand",
    "stance_too_high",
    "incomplete_arm_extension",
    "pumping_the_arm",
    "over_parrying",
    "wide_step",
    "narrow_step",
    "center_of_mass_in_front",
    "center_of_mass_leaning_backward",
]


@dataclass
class HeuristicMetric:
    heuristic_key: str
    triggered: bool
    primary_value: Optional[float]
    threshold: str
    values: Dict[str, Any]


def format_value(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def format_metrics(metric: HeuristicMetric) -> str:
    parts = []
    for key, value in metric.values.items():
        parts.append(f"{key}={format_value(value)}")
    if metric.threshold:
        parts.append(f"threshold={metric.threshold}")
    return " | ".join(parts)


def compute_heuristic_metric(
    heuristic_key: str,
    skeletons: List[Dict[str, Any]],
    target_side: str = "left",
    training_mode: str = "Free Bouting",
) -> HeuristicMetric:
    if not skeletons:
        return _empty_metric(heuristic_key, "no skeletons")

    if heuristic_key == "bounce_excessive":
        return _metric_bounce(skeletons)
    if heuristic_key == "lunge_overextension":
        return _metric_lunge(skeletons, target_side)
    if heuristic_key == "guard_dropped":
        return _metric_guard(skeletons, target_side, training_mode)
    if heuristic_key == "foot_before_hand":
        return _metric_foot_before_hand(skeletons, target_side)
    if heuristic_key == "stance_too_high":
        return _metric_stance_too_high(skeletons, target_side)
    if heuristic_key == "incomplete_arm_extension":
        return _metric_incomplete_arm_extension(skeletons, target_side)
    if heuristic_key == "pumping_the_arm":
        return _metric_pumping_the_arm(skeletons, target_side)
    if heuristic_key == "over_parrying":
        return _metric_over_parrying(skeletons, target_side)
    if heuristic_key in {"wide_step", "narrow_step"}:
        return _metric_step_width(skeletons, target_side, heuristic_key)
    if heuristic_key in {"center_of_mass_in_front", "center_of_mass_leaning_backward"}:
        return _metric_center_of_mass(skeletons, target_side, heuristic_key)
    return _empty_metric(heuristic_key, "unsupported heuristic")


def build_debug_events(
    report: Dict[str, Any],
    heuristic_key: str,
    target_side: str,
    training_mode: str,
    fps: float,
) -> List[Dict[str, Any]]:
    fps = fps or 30.0
    skeleton_by_frame = _target_skeletons_by_frame(report)
    errors = report.get("posture_errors", [])
    action_segments = report.get("action_segments", [])
    if not action_segments and skeleton_by_frame:
        frames = sorted(skeleton_by_frame)
        action_segments = [{
            "action": "Unclassified",
            "video_start_frame": frames[0],
            "video_end_frame": frames[-1],
        }]
    rows: List[Dict[str, Any]] = []

    keys = HEURISTIC_KEYS if heuristic_key == "all" else [heuristic_key]
    for seg_idx, seg in enumerate(action_segments):
        start = int(seg.get("video_start_frame", seg.get("start_frame", 0)))
        end = int(seg.get("video_end_frame", seg.get("end_frame", start)))
        window_skeletons = [
            skeleton_by_frame[i]
            for i in range(start, end + 1)
            if i in skeleton_by_frame and skeleton_by_frame[i]
        ]
        segment_errors = [
            err for err in errors
            if _error_overlaps_segment(err, seg_idx, start, end)
        ]

        for key in keys:
            metric = compute_heuristic_metric(
                key,
                window_skeletons,
                target_side=target_side,
                training_mode=training_mode,
            )
            alert_errors = [
                err for err in segment_errors
                if err.get("error_key", err.get("error")) == key
            ]
            rows.append({
                "segment_index": seg_idx,
                "heuristic": key,
                "action": seg.get("action", ""),
                "start_frame": start,
                "end_frame": end,
                "time": f"{start / fps:.2f}s - {end / fps:.2f}s",
                "alert": bool(alert_errors),
                "metric_triggered": metric.triggered,
                "primary_value": metric.primary_value,
                "threshold": metric.threshold,
                "metric_values": metric.values,
                "metrics": format_metrics(metric),
                "alert_time": _alert_time(alert_errors, fps),
            })

    return rows


def build_frame_debug_events(
    report: Dict[str, Any],
    heuristic_key: str,
    target_side: str,
    training_mode: str,
    fps: float,
    window_size: int = 28,
    frame_step: int = 1,
) -> List[Dict[str, Any]]:
    """Build one heuristic row per target frame using a rolling skeleton window."""
    fps = fps or 30.0
    window_size = max(1, int(window_size))
    frame_step = max(1, int(frame_step))
    skeleton_by_frame = _target_skeletons_by_frame(report)
    frame_numbers = sorted(skeleton_by_frame)
    if not frame_numbers:
        return []

    errors = report.get("posture_errors", [])
    action_segments = report.get("action_segments", [])
    keys = HEURISTIC_KEYS if heuristic_key == "all" else [heuristic_key]
    rows: List[Dict[str, Any]] = []

    for sample_index, frame_idx in enumerate(frame_numbers):
        if sample_index % frame_step != 0:
            continue
        window_frames = frame_numbers[max(0, sample_index - window_size + 1):sample_index + 1]
        window_skeletons = [skeleton_by_frame[i] for i in window_frames if skeleton_by_frame.get(i)]
        if not window_skeletons:
            continue

        start = window_frames[0]
        end = frame_idx
        action = _action_for_frame(action_segments, frame_idx)

        for key in keys:
            metric = compute_heuristic_metric(
                key,
                window_skeletons,
                target_side=target_side,
                training_mode=training_mode,
            )
            alert_errors = [
                err for err in errors
                if err.get("error_key", err.get("error")) == key
                and _error_overlaps_frame(err, frame_idx)
            ]
            metric_values = dict(metric.values)
            metric_values["sample_count"] = len(window_skeletons)
            metric_values["window_size"] = window_size
            rows.append({
                "frame_index": frame_idx,
                "segment_index": "",
                "heuristic": key,
                "action": action,
                "start_frame": start,
                "end_frame": end,
                "time": f"{frame_idx / fps:.2f}s",
                "window": f"{start}-{end}",
                "alert": bool(alert_errors),
                "metric_triggered": metric.triggered,
                "primary_value": metric.primary_value,
                "threshold": metric.threshold,
                "metric_values": metric_values,
                "metrics": format_metrics(metric),
                "alert_time": _alert_time(alert_errors, fps),
            })

    return rows


def _empty_metric(heuristic_key: str, reason: str) -> HeuristicMetric:
    return HeuristicMetric(
        heuristic_key=heuristic_key,
        triggered=False,
        primary_value=None,
        threshold="n/a",
        values={"status": reason},
    )


def _metric_bounce(skeletons: List[Dict[str, Any]]) -> HeuristicMetric:
    pelvis_ys: List[float] = []
    all_ys: List[float] = []
    for skel in skeletons:
        pc = _pelvis_center(skel)
        if pc is not None:
            pelvis_ys.append(float(pc[1]))
        for joint in skel.values():
            if isinstance(joint, (list, tuple, np.ndarray)) and len(joint) == 2:
                all_ys.append(float(joint[1]))

    if len(pelvis_ys) < 5 or len(all_ys) < 2:
        return _empty_metric("bounce_excessive", "need >=5 pelvis samples")
    bbox_height = max(all_ys) - min(all_ys)
    if bbox_height < 1e-6:
        return _empty_metric("bounce_excessive", "bbox height too small")
    delta_y = max(pelvis_ys) - min(pelvis_ys)
    ratio = delta_y / bbox_height
    return HeuristicMetric(
        "bounce_excessive",
        ratio > 0.10,
        ratio,
        "pelvis_delta / bbox_height > 0.10",
        {"pelvis_delta": delta_y, "bbox_height": bbox_height, "ratio": ratio},
    )


def _metric_lunge(skeletons: List[Dict[str, Any]], target_side: str) -> HeuristicMetric:
    limbs = FRONT_LIMBS[target_side]
    ref_ankle = _get_joint(skeletons[0], limbs["ankle"])
    if ref_ankle is None:
        return _empty_metric("lunge_overextension", "missing front ankle")

    max_disp = 0.0
    peak_skel = skeletons[0]
    peak_index = 0
    for i, skel in enumerate(skeletons):
        ankle = _get_joint(skel, limbs["ankle"])
        if ankle is not None:
            disp = float(np.linalg.norm(ankle - ref_ankle))
            if disp > max_disp:
                max_disp = disp
                peak_skel = skel
                peak_index = i

    hip = _get_joint(peak_skel, limbs["hip"])
    knee = _get_joint(peak_skel, limbs["knee"])
    ankle = _get_joint(peak_skel, limbs["ankle"])
    if hip is None or knee is None or ankle is None:
        return _empty_metric("lunge_overextension", "missing front leg joints")
    angle = calc_angle(hip, knee, ankle)
    return HeuristicMetric(
        "lunge_overextension",
        angle < 90.0,
        angle,
        "front_knee_angle < 90 deg",
        {"front_knee_angle": angle, "front_ankle_disp": max_disp, "peak_index": peak_index},
    )


def _metric_guard(
    skeletons: List[Dict[str, Any]],
    target_side: str,
    training_mode: str,
) -> HeuristicMetric:
    limbs = FRONT_LIMBS[target_side]
    threshold = 20 if training_mode == "Free Bouting" else 10
    consecutive = 0
    max_consecutive = 0
    frames_low = 0
    for skel in skeletons:
        wrist = _get_joint(skel, limbs["wrist"])
        pelvis = _pelvis_center(skel)
        is_low = wrist is not None and pelvis is not None and wrist[1] > pelvis[1]
        if is_low:
            frames_low += 1
            consecutive += 1
            max_consecutive = max(max_consecutive, consecutive)
        else:
            consecutive = 0
    return HeuristicMetric(
        "guard_dropped",
        max_consecutive > threshold,
        float(max_consecutive),
        f"max_consecutive_low_guard > {threshold} frames",
        {"max_consecutive": max_consecutive, "low_frames": frames_low, "threshold_frames": threshold},
    )


def _metric_foot_before_hand(skeletons: List[Dict[str, Any]], target_side: str) -> HeuristicMetric:
    limbs = FRONT_LIMBS[target_side]
    ref_wrist = _get_joint(skeletons[0], limbs["wrist"])
    ref_ankle = _get_joint(skeletons[0], limbs["ankle"])
    if ref_wrist is None or ref_ankle is None:
        return _empty_metric("foot_before_hand", "missing wrist or ankle")

    max_wrist_disp = 0.0
    max_ankle_disp = 0.0
    wrist_peak = 0
    ankle_peak = 0
    for i, skel in enumerate(skeletons):
        wrist = _get_joint(skel, limbs["wrist"])
        ankle = _get_joint(skel, limbs["ankle"])
        if wrist is not None:
            disp = abs(float(wrist[0] - ref_wrist[0]))
            if disp > max_wrist_disp:
                max_wrist_disp = disp
                wrist_peak = i
        if ankle is not None:
            disp = abs(float(ankle[0] - ref_ankle[0]))
            if disp > max_ankle_disp:
                max_ankle_disp = disp
                ankle_peak = i
    triggered = max_ankle_disp > 5 and max_wrist_disp > 5 and ankle_peak < wrist_peak
    return HeuristicMetric(
        "foot_before_hand",
        triggered,
        float(ankle_peak - wrist_peak),
        "ankle_disp > 5, wrist_disp > 5, ankle_peak_frame < wrist_peak_frame",
        {
            "wrist_peak_index": wrist_peak,
            "ankle_peak_index": ankle_peak,
            "wrist_disp": max_wrist_disp,
            "ankle_disp": max_ankle_disp,
        },
    )


def _metric_stance_too_high(skeletons: List[Dict[str, Any]], target_side: str) -> HeuristicMetric:
    limbs = FRONT_LIMBS[target_side]
    angles = []
    for skel in skeletons:
        hip = _get_joint(skel, limbs["hip"])
        knee = _get_joint(skel, limbs["knee"])
        ankle = _get_joint(skel, limbs["ankle"])
        if hip is not None and knee is not None and ankle is not None:
            angles.append(calc_angle(hip, knee, ankle))
    if len(angles) < 3:
        return _empty_metric("stance_too_high", "need >=3 knee angles")
    avg_angle = float(np.mean(angles))
    return HeuristicMetric(
        "stance_too_high",
        avg_angle > 160.0,
        avg_angle,
        "avg_front_knee_angle > 160 deg",
        {"avg_front_knee_angle": avg_angle, "min_angle": min(angles), "max_angle": max(angles)},
    )


def _metric_incomplete_arm_extension(
    skeletons: List[Dict[str, Any]],
    target_side: str,
) -> HeuristicMetric:
    limbs = FRONT_LIMBS[target_side]
    ref_wrist = _get_joint(skeletons[0], limbs["wrist"])
    if ref_wrist is None:
        return _empty_metric("incomplete_arm_extension", "missing wrist")

    max_disp = 0.0
    peak_skel = skeletons[0]
    peak_index = 0
    for i, skel in enumerate(skeletons):
        wrist = _get_joint(skel, limbs["wrist"])
        if wrist is not None:
            disp = abs(float(wrist[0] - ref_wrist[0]))
            if disp > max_disp:
                max_disp = disp
                peak_skel = skel
                peak_index = i

    shoulder = _get_joint(peak_skel, limbs["shoulder"])
    elbow = _get_joint(peak_skel, limbs["elbow"])
    wrist = _get_joint(peak_skel, limbs["wrist"])
    if shoulder is None or elbow is None or wrist is None:
        return _empty_metric("incomplete_arm_extension", "missing arm joints")
    angle = calc_angle(shoulder, elbow, wrist)
    return HeuristicMetric(
        "incomplete_arm_extension",
        angle < 155.0,
        angle,
        "arm_angle_at_peak_extension < 155 deg",
        {"arm_angle": angle, "wrist_disp": max_disp, "peak_index": peak_index},
    )


def _metric_pumping_the_arm(skeletons: List[Dict[str, Any]], target_side: str) -> HeuristicMetric:
    limbs = FRONT_LIMBS[target_side]
    if len(skeletons) < 6:
        return _empty_metric("pumping_the_arm", "need >=6 frames")
    first = _get_joint(skeletons[0], limbs["wrist"])
    last = _get_joint(skeletons[-1], limbs["wrist"])
    if first is None or last is None:
        return _empty_metric("pumping_the_arm", "missing wrist")
    overall_dx = float(last[0] - first[0])
    if abs(overall_dx) < 10:
        return _empty_metric("pumping_the_arm", "overall wrist dx < 10")

    attack_dir = 1.0 if overall_dx > 0 else -1.0
    early_end = max(2, len(skeletons) // 3)
    min_retract = 0.0
    for skel in skeletons[:early_end]:
        wrist = _get_joint(skel, limbs["wrist"])
        if wrist is not None:
            dx = float(wrist[0] - first[0]) * attack_dir
            min_retract = min(min_retract, dx)
    return HeuristicMetric(
        "pumping_the_arm",
        min_retract < -8.0,
        min_retract,
        "early_wrist_retraction < -8 px",
        {"overall_dx": overall_dx, "early_min_retract": min_retract, "early_frames": early_end},
    )


def _metric_over_parrying(skeletons: List[Dict[str, Any]], target_side: str) -> HeuristicMetric:
    limbs = FRONT_LIMBS[target_side]
    shoulder_width = None
    for skel in skeletons:
        shoulder = _get_joint(skel, limbs["shoulder"])
        other_name = "left_shoulder" if target_side == "right" else "right_shoulder"
        other = _get_joint(skel, other_name)
        if other is not None and shoulder is not None:
            shoulder_width = abs(float(shoulder[0] - other[0]))
            break
        pelvis = _pelvis_center(skel)
        if shoulder is not None and pelvis is not None:
            shoulder_width = abs(float(shoulder[0] - pelvis[0])) * 2.0
            break
    if shoulder_width is None or shoulder_width < 1e-6:
        return _empty_metric("over_parrying", "missing shoulder width")

    wrist_xs = []
    for skel in skeletons:
        wrist = _get_joint(skel, limbs["wrist"])
        if wrist is not None:
            wrist_xs.append(float(wrist[0]))
    if len(wrist_xs) < 5:
        return _empty_metric("over_parrying", "need >=5 wrist samples")
    sweep = max(wrist_xs) - min(wrist_xs)
    ratio = sweep / shoulder_width
    return HeuristicMetric(
        "over_parrying",
        ratio > 2.0,
        ratio,
        "wrist_sweep / shoulder_width > 2.0",
        {"wrist_sweep": sweep, "shoulder_width": shoulder_width, "ratio": ratio},
    )


def _metric_step_width(
    skeletons: List[Dict[str, Any]],
    target_side: str,
    heuristic_key: str,
) -> HeuristicMetric:
    ratios = []
    limbs = FRONT_LIMBS[target_side]
    for skel in skeletons:
        front_ankle = _get_joint(skel, limbs["ankle"])
        back_ankle_name = "left_ankle" if target_side == "right" else "right_ankle"
        back_ankle = _get_joint(skel, back_ankle_name)
        front_shoulder = _get_joint(skel, limbs["shoulder"])
        pelvis = _pelvis_center(skel)
        if front_ankle is None or back_ankle is None or front_shoulder is None or pelvis is None:
            continue
        shoulder_width = abs(float(front_shoulder[0] - pelvis[0])) * 2.0
        if shoulder_width < 10.0:
            continue
        step_width = abs(float(front_ankle[0] - back_ankle[0]))
        ratios.append(step_width / shoulder_width)
    if not ratios:
        return _empty_metric(heuristic_key, "missing step-width samples")

    primary = max(ratios) if heuristic_key == "wide_step" else min(ratios)
    triggered = primary > 3.0 if heuristic_key == "wide_step" else primary < 1.0
    threshold = "max_ratio > 3.0" if heuristic_key == "wide_step" else "min_ratio < 1.0"
    return HeuristicMetric(
        heuristic_key,
        triggered,
        primary,
        threshold,
        {"min_ratio": min(ratios), "max_ratio": max(ratios), "samples": len(ratios)},
    )


def _metric_center_of_mass(
    skeletons: List[Dict[str, Any]],
    target_side: str,
    heuristic_key: str,
) -> HeuristicMetric:
    ratios = []
    limbs = FRONT_LIMBS[target_side]
    for skel in skeletons:
        front_ankle = _get_joint(skel, limbs["ankle"])
        back_ankle_name = "left_ankle" if target_side == "right" else "right_ankle"
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
        if front_x > back_x:
            ratio = (pelvis_x - back_x) / base_width
        else:
            ratio = (back_x - pelvis_x) / base_width
        ratios.append(ratio)
    if not ratios:
        return _empty_metric(heuristic_key, "missing center-of-mass samples")

    primary = max(ratios) if heuristic_key == "center_of_mass_in_front" else min(ratios)
    triggered = primary > 0.65 if heuristic_key == "center_of_mass_in_front" else primary < 0.35
    threshold = "max_ratio > 0.65" if heuristic_key == "center_of_mass_in_front" else "min_ratio < 0.35"
    return HeuristicMetric(
        heuristic_key,
        triggered,
        primary,
        threshold,
        {"min_ratio": min(ratios), "max_ratio": max(ratios), "samples": len(ratios)},
    )


def _target_skeletons_by_frame(report: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    tracking = report.get("two_fencer_tracking", {})
    locked_track_id = tracking.get("locked_track_id")
    result = {}
    for frame in tracking.get("frames", []):
        frame_index = frame.get("frame_index")
        if frame_index is None:
            continue
        if frame.get("target_skeleton"):
            result[int(frame_index)] = frame["target_skeleton"]
            continue
        target = _target_detection(frame.get("tracks", []), locked_track_id)
        if target and target.get("skeleton"):
            result[int(frame_index)] = target["skeleton"]
    return result


def _target_detection(
    tracks: Iterable[Dict[str, Any]],
    locked_track_id: Optional[int],
) -> Optional[Dict[str, Any]]:
    tracks = list(tracks)
    if not tracks:
        return None
    if locked_track_id is not None:
        for det in tracks:
            if det.get("track_id") == locked_track_id:
                return det
    return max(tracks, key=lambda det: det.get("area", 0))


def _error_overlaps_segment(
    err: Dict[str, Any],
    segment_index: int,
    start: int,
    end: int,
) -> bool:
    if err.get("segment_index") == segment_index:
        return True
    err_start = int(err.get("start_frame", -1))
    err_end = int(err.get("end_frame", err_start))
    return err_start <= end and err_end >= start


def _error_overlaps_frame(err: Dict[str, Any], frame_idx: int) -> bool:
    err_start = int(err.get("start_frame", -1))
    err_end = int(err.get("end_frame", err_start))
    return err_start <= frame_idx <= err_end


def _action_for_frame(action_segments: List[Dict[str, Any]], frame_idx: int) -> str:
    for seg in action_segments:
        start = int(seg.get("video_start_frame", seg.get("start_frame", -1)))
        end = int(seg.get("video_end_frame", seg.get("end_frame", start)))
        if start <= frame_idx <= end:
            return seg.get("action", "")
    return ""


def _alert_time(errors: List[Dict[str, Any]], fps: float) -> str:
    if not errors:
        return ""
    starts = [int(err.get("start_frame", 0)) for err in errors]
    return ", ".join(f"{frame / fps:.2f}s" for frame in starts)
