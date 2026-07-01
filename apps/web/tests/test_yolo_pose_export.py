from pathlib import Path

import pytest


ort = pytest.importorskip("onnxruntime")


def test_android_yolo_pose_asset_shape():
    asset = Path("android/app/src/main/assets/yolo_pose.onnx")
    assert asset.exists()

    session = ort.InferenceSession(str(asset), providers=["CPUExecutionProvider"])
    inputs = session.get_inputs()
    outputs = session.get_outputs()

    assert len(inputs) == 1
    assert inputs[0].name == "images"
    assert inputs[0].shape == [1, 3, 640, 640]
    assert len(outputs) == 1
    assert outputs[0].shape == [1, 56, 8400]
