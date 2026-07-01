import numpy as np
import pytest
import torch

from inference.activity_gatekeeper import ActivityGatekeeper
from inference.heuristics_engine import HeuristicsEngine
from inference.sliding_window import SlidingWindowInference
from inference.target_tracker import TargetTracker
from src.models.fencenet_v2 import FenceNetV2
from src.pose_estimation import PoseEstimator


def _raw_skeleton():
    return {
        "nose": (100.0, 20.0),
        "left_shoulder": (80.0, 50.0),
        "right_shoulder": (120.0, 50.0),
        "left_elbow": (70.0, 70.0),
        "right_elbow": (130.0, 70.0),
        "left_wrist": (60.0, 90.0),
        "right_wrist": (140.0, 90.0),
        "left_hip": (85.0, 110.0),
        "right_hip": (115.0, 110.0),
        "left_knee": (80.0, 150.0),
        "right_knee": (120.0, 150.0),
        "left_ankle": (75.0, 190.0),
        "right_ankle": (125.0, 190.0),
    }


def _tracking_detection(center_x, track_id=None):
    skeleton = {
        "nose": (center_x, 20.0),
        "front_wrist": (center_x + 20.0, 70.0),
        "left_hip": (center_x - 5.0, 100.0),
        "right_hip": (center_x + 5.0, 100.0),
        "left_knee": (center_x - 5.0, 140.0),
        "right_knee": (center_x + 5.0, 140.0),
        "left_ankle": (center_x - 8.0, 180.0),
        "right_ankle": (center_x + 8.0, 180.0),
    }
    item = {
        "bbox": [center_x - 40.0, 0.0, center_x + 40.0, 200.0],
        "area": 16000.0,
        "center": [center_x, 100.0],
        "confidence": 1.0,
        "skeleton": skeleton,
    }
    if track_id is not None:
        item["track_id"] = track_id
    return item


def _gate_skeleton(knee_angle="bent", shift=0.0):
    if knee_angle == "bent":
        hip = (100.0 + shift, 100.0)
        knee = (100.0 + shift, 140.0)
        ankle = (140.0 + shift, 140.0)
    else:
        hip = (100.0 + shift, 100.0)
        knee = (100.0 + shift, 140.0)
        ankle = (100.0 + shift, 180.0)
    return {
        "nose": (100.0 + shift, 40.0),
        "left_shoulder": (80.0 + shift, 70.0),
        "right_shoulder": (120.0 + shift, 70.0),
        "front_shoulder": (120.0 + shift, 70.0),
        "front_elbow": (130.0 + shift, 85.0),
        "front_wrist": (140.0 + shift, 95.0),
        "left_hip": (85.0 + shift, 100.0),
        "right_hip": hip,
        "left_knee": (85.0 + shift, 140.0),
        "right_knee": knee,
        "left_ankle": (85.0 + shift, 180.0),
        "right_ankle": ankle,
    }


def test_canonical_skeleton_uses_target_side_by_default():
    left_target = PoseEstimator.canonicalize_skeleton(_raw_skeleton(), target_side="left")
    right_target = PoseEstimator.canonicalize_skeleton(_raw_skeleton(), target_side="right")

    assert left_target["front_wrist"] == _raw_skeleton()["right_wrist"]
    assert left_target["front_ankle"] == _raw_skeleton()["right_ankle"]
    assert right_target["front_wrist"] == _raw_skeleton()["left_wrist"]
    assert right_target["front_ankle"] == _raw_skeleton()["left_ankle"]


def test_canonical_skeleton_handedness_override_preserves_compatibility():
    skeleton = PoseEstimator.canonicalize_skeleton(
        _raw_skeleton(),
        target_side="right",
        handedness="right",
    )

    assert skeleton["front_wrist"] == _raw_skeleton()["right_wrist"]
    assert skeleton["front_ankle"] == _raw_skeleton()["right_ankle"]


def test_tracker_predicts_short_missing_gap_instead_of_repeating_last_pose():
    tracker = TargetTracker(target_side="left")
    tracker.process_frame_detections(
        [_tracking_detection(100.0, track_id=1), _tracking_detection(800.0, track_id=2)],
        frame_idx=0,
    )
    tracker.process_frame_detections(
        [_tracking_detection(110.0, track_id=1), _tracking_detection(800.0, track_id=2)],
        frame_idx=1,
    )

    target, _ = tracker.process_frame_detections([], frame_idx=2)

    assert target["nose"][0] == 120.0
    assert tracker.last_interpolated is True
    assert tracker.lock_state == "interpolating"


def test_gatekeeper_activates_and_idles_with_knee_hysteresis():
    gatekeeper = ActivityGatekeeper(fps=5)
    gatekeeper.idle_trigger_threshold = 2

    for frame in range(gatekeeper.active_trigger_threshold):
        is_active = gatekeeper.update(
            _gate_skeleton("bent", shift=float(frame)),
            None,
            frame_width=640,
            target_side="left",
        )

    assert is_active is True
    assert gatekeeper.state == gatekeeper.STATE_ACTIVE

    for frame in range(gatekeeper.idle_trigger_threshold):
        is_active = gatekeeper.update(
            _gate_skeleton("standing", shift=10.0 + frame),
            None,
            frame_width=640,
            target_side="left",
        )

    assert is_active is False
    assert gatekeeper.state == gatekeeper.STATE_IDLE
    assert gatekeeper.last_reasons["standing_up"] is True


def test_realtime_idle_frames_do_not_call_fencenet_classifier():
    from src.realtime.realtime_app import LiveVideoPipeline

    pipeline = LiveVideoPipeline(pose_backend="mock", voice_enabled=False)
    pipeline.sliding_window.classify_window = pytest.fail
    frame = np.zeros((240, 320, 3), dtype=np.uint8)

    for _ in range(35):
        pipeline.process_frame(frame.copy(), draw_hud=False)

    assert pipeline.current_action == "Idle"
    assert len(pipeline.normalized_skeletons) == 0


def test_checkpoint_metadata_mismatch_marks_model_unavailable(tmp_path):
    checkpoint_path = tmp_path / "bad_checkpoint.pt"
    model = FenceNetV2(input_channels=18)
    torch.save(
        {
            "metadata": {
                "input_channels": 20,
                "num_classes": 6,
                "action_classes": model.get_class_names(),
                "sequence_length": 28,
            },
            "state_dict": model.state_dict(),
        },
        checkpoint_path,
    )

    classifier = SlidingWindowInference(model_path=str(checkpoint_path), device="cpu")

    assert classifier.model_loaded is False
    with pytest.raises(RuntimeError, match="input_channels mismatch"):
        classifier.classify_window(np.zeros((28, 9, 2), dtype=np.float32))


def test_nms_merges_overlapping_step_windows_not_only_attacks():
    classifier = SlidingWindowInference(model_path=None, device="cpu")
    merged = classifier._nms([
        {"start_frame": 0, "end_frame": 28, "action": "SF", "confidence": 0.7},
        {"start_frame": 10, "end_frame": 38, "action": "SF", "confidence": 0.9},
    ])

    assert len(merged) == 1
    assert merged[0]["action"] == "SF"
    assert merged[0]["confidence"] == 0.9
    assert merged[0]["merged_window_count"] == 2


def test_heuristic_errors_include_metric_evidence_and_severity():
    skeletons = [
        {
            **_gate_skeleton("bent"),
            "front_wrist": (140.0, 150.0),
        }
        for _ in range(11)
    ]
    engine = HeuristicsEngine(target_side="left", training_mode="Footwork")

    errors = engine.evaluate(
        [{"start_frame": 0, "end_frame": len(skeletons), "action": "SF"}],
        skeletons,
    )
    guard = next(error for error in errors if error["error_key"] == "guard_dropped")

    assert guard["metric_name"] == "consecutive_frames"
    assert guard["metric_value"] > guard["threshold"]
    assert guard["evidence_frame"] == 10
    assert guard["sample_count"] == 11
    assert guard["severity"] == "medium"
