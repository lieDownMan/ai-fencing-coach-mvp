from inference.heuristic_debug import (
    build_debug_events,
    build_frame_debug_events,
    compute_heuristic_metric,
)
from inference.heuristics_engine import BOUNCE_RATIO_THRESHOLD


def _skeleton(pelvis_y=100.0, wrist_y=80.0):
    return {
        "nose": (100.0, 30.0),
        "front_shoulder": (100.0, 60.0),
        "front_elbow": (115.0, 70.0),
        "front_wrist": (130.0, wrist_y),
        "left_hip": (90.0, pelvis_y),
        "right_hip": (110.0, pelvis_y),
        "left_knee": (85.0, 150.0),
        "right_knee": (115.0, 150.0),
        "left_ankle": (80.0, 200.0),
        "right_ankle": (120.0, 200.0),
    }


def test_bounce_metric_reports_ratio_and_trigger():
    skeletons = [_skeleton(pelvis_y=y) for y in [100, 120, 140, 160, 180]]

    metric = compute_heuristic_metric("bounce_excessive", skeletons)

    assert metric.triggered is True
    assert metric.values["pelvis_delta"] == 80.0
    assert metric.primary_value > BOUNCE_RATIO_THRESHOLD


def test_guard_metric_uses_training_mode_threshold():
    skeletons = [_skeleton(wrist_y=130.0) for _ in range(11)]

    metric = compute_heuristic_metric(
        "guard_dropped",
        skeletons,
        target_side="left",
        training_mode="Footwork",
    )

    assert metric.triggered is True
    assert metric.values["max_consecutive"] == 11
    assert metric.values["threshold_frames"] == 10


def test_build_debug_events_marks_system_alert_and_metric_value():
    frames = []
    for i, pelvis_y in enumerate([100, 120, 140, 160, 180]):
        frames.append({
            "frame_index": i,
            "tracks": [{
                "track_id": 7,
                "area": 1000,
                "skeleton": _skeleton(pelvis_y=pelvis_y),
            }],
        })
    report = {
        "two_fencer_tracking": {
            "locked_track_id": 7,
            "frames": frames,
        },
        "action_segments": [{
            "action": "SF",
            "video_start_frame": 0,
            "video_end_frame": 4,
        }],
        "posture_errors": [{
            "error_key": "bounce_excessive",
            "segment_index": 0,
            "start_frame": 0,
            "end_frame": 4,
        }],
    }

    rows = build_debug_events(
        report,
        heuristic_key="bounce_excessive",
        target_side="left",
        training_mode="Footwork",
        fps=10.0,
    )

    assert len(rows) == 1
    assert rows[0]["alert"] is True
    assert rows[0]["metric_triggered"] is True
    assert rows[0]["alert_time"] == "0.00s"
    assert rows[0]["primary_value"] > BOUNCE_RATIO_THRESHOLD
    assert rows[0]["metric_values"]["ratio"] > BOUNCE_RATIO_THRESHOLD


def test_build_debug_events_falls_back_to_unclassified_range():
    report = {
        "two_fencer_tracking": {
            "locked_track_id": None,
            "frames": [
                {
                    "frame_index": i,
                    "tracks": [{"area": 1000, "skeleton": _skeleton()}],
                }
                for i in range(5)
            ],
        },
        "action_segments": [],
        "posture_errors": [],
    }

    rows = build_debug_events(
        report,
        heuristic_key="stance_too_high",
        target_side="left",
        training_mode="Footwork",
        fps=30.0,
    )

    assert len(rows) == 1
    assert rows[0]["action"] == "Unclassified"
    assert rows[0]["start_frame"] == 0
    assert rows[0]["end_frame"] == 4


def test_build_frame_debug_events_returns_one_row_per_frame_and_heuristic():
    report = {
        "two_fencer_tracking": {
            "locked_track_id": None,
            "frames": [
                {
                    "frame_index": i,
                    "target_skeleton": _skeleton(pelvis_y=100 + i * 20),
                    "tracks": [],
                }
                for i in range(5)
            ],
        },
        "action_segments": [{
            "action": "SF",
            "video_start_frame": 0,
            "video_end_frame": 4,
        }],
        "posture_errors": [{
            "error_key": "bounce_excessive",
            "segment_index": 0,
            "start_frame": 4,
            "end_frame": 4,
        }],
    }

    rows = build_frame_debug_events(
        report,
        heuristic_key="bounce_excessive",
        target_side="left",
        training_mode="Footwork",
        fps=10.0,
        window_size=5,
    )

    assert len(rows) == 5
    assert rows[-1]["frame_index"] == 4
    assert rows[-1]["time"] == "0.40s"
    assert rows[-1]["window"] == "0-4"
    assert rows[-1]["action"] == "SF"
    assert rows[-1]["alert"] is True
    assert rows[-1]["metric_triggered"] is True
    assert rows[-1]["metric_values"]["sample_count"] == 5
