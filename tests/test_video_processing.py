import cv2
import numpy as np

from inference.sliding_window import scale_pose_detections
from inference.video_annotator import VideoAnnotator


def _write_video(path, size=(320, 240), frame_count=3):
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        10.0,
        size,
    )
    assert writer.isOpened()
    for index in range(frame_count):
        frame = np.full((size[1], size[0], 3), index * 30, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def test_scale_pose_detections_restores_original_geometry():
    detections = [
        {
            "bbox": [10, 20, 110, 220],
            "center": [60, 120],
            "area": 20000,
            "skeleton": {
                "nose": (50, 40),
                "front_wrist": (80, 100),
            },
        }
    ]

    scaled = scale_pose_detections(detections, scale_x=2.0, scale_y=3.0)

    assert scaled[0]["bbox"] == [20.0, 60.0, 220.0, 660.0]
    assert scaled[0]["center"] == [120.0, 360.0]
    assert scaled[0]["area"] == 120000.0
    assert scaled[0]["skeleton"]["front_wrist"] == (160.0, 300.0)


def test_video_annotator_can_write_downscaled_output(tmp_path):
    input_path = tmp_path / "input.mp4"
    output_path = tmp_path / "output.mp4"
    _write_video(input_path, size=(320, 240), frame_count=3)

    VideoAnnotator().annotate_video(
        str(input_path),
        str(output_path),
        {"two_fencer_tracking": {"frames": []}, "action_segments": [], "posture_errors": []},
        max_width=160,
    )

    cap = cv2.VideoCapture(str(output_path))
    assert cap.isOpened()
    assert int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) == 160
    assert int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) == 120
    cap.release()
