from inference.target_tracker import TargetTracker


def _skeleton(label_x):
    return {
        "nose": (label_x, 10.0),
        "front_shoulder": (label_x, 30.0),
        "front_elbow": (label_x, 45.0),
        "front_wrist": (label_x, 60.0),
        "left_hip": (label_x - 5.0, 80.0),
        "right_hip": (label_x + 5.0, 80.0),
        "left_knee": (label_x - 5.0, 120.0),
        "right_knee": (label_x + 5.0, 120.0),
        "left_ankle": (label_x - 5.0, 160.0),
        "right_ankle": (label_x + 5.0, 160.0),
    }


def _det(center_x, source_rank=0, track_id=None):
    bbox = [center_x - 50.0, 0.0, center_x + 50.0, 200.0]
    det = {
        "bbox": bbox,
        "area": 20000.0,
        "source_rank": source_rank,
        "skeleton": _skeleton(center_x),
    }
    if track_id is not None:
        det["track_id"] = track_id
    return det


def test_position_fallback_does_not_follow_flipped_source_rank():
    tracker = TargetTracker(target_side="left")

    target, _ = tracker.process_frame_detections(
        [_det(150, source_rank=0), _det(850, source_rank=1)],
        frame_idx=0,
    )
    assert target["nose"][0] == 150

    target, _ = tracker.process_frame_detections(
        [_det(850, source_rank=0), _det(165, source_rank=1)],
        frame_idx=1,
    )

    assert target["nose"][0] == 165


def test_position_fallback_does_not_switch_to_far_opponent_when_target_missing():
    tracker = TargetTracker(target_side="left")
    first_target, _ = tracker.process_frame_detections(
        [_det(150, source_rank=0), _det(850, source_rank=1)],
        frame_idx=0,
    )

    target, _ = tracker.process_frame_detections(
        [_det(850, source_rank=0)],
        frame_idx=1,
    )

    assert target == first_target
    assert tracker.missing_frames_count == 1


def test_position_fallback_recovers_when_track_id_changes_near_last_target():
    tracker = TargetTracker(target_side="left")
    target, _ = tracker.process_frame_detections(
        [_det(150, track_id=10), _det(850, track_id=20)],
        frame_idx=0,
    )
    assert target["nose"][0] == 150
    assert tracker.locked_track_id == 10

    target, _ = tracker.process_frame_detections(
        [_det(165, track_id=30), _det(850, track_id=20)],
        frame_idx=1,
    )

    assert target["nose"][0] == 165
    assert tracker.locked_track_id == 30


def test_tracker_rejects_implausible_track_id_jump():
    tracker = TargetTracker(target_side="left")
    tracker.process_frame_detections(
        [_det(150, track_id=10), _det(850, track_id=20)],
        frame_idx=0,
    )

    target, _ = tracker.process_frame_detections(
        [_det(850, track_id=10), _det(165, track_id=30)],
        frame_idx=1,
    )

    assert target["nose"][0] == 165
    assert tracker.locked_track_id == 30
