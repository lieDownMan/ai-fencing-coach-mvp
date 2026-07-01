from pathlib import Path

import pytest


torch = pytest.importorskip("torch")
onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")

from scripts.export_fencenet_onnx import (  # noqa: E402
    INPUT_NAME,
    OUTPUT_NAME,
    build_model,
    check_onnx_against_torch,
    export_model,
)
from src.data.fencing_dataset import NUM_CHANNELS, NUM_CLASSES, SEQUENCE_LENGTH  # noqa: E402


def test_exported_fencenet_onnx_shape_and_names(tmp_path: Path):
    torch.manual_seed(7)
    model = build_model(None, allow_random=True)
    output = export_model(model, tmp_path / "fencenet_v2.onnx")

    exported = onnx.load(str(output))
    assert exported.graph.input[0].name == INPUT_NAME
    assert exported.graph.output[0].name == OUTPUT_NAME

    input_dims = [dim.dim_value for dim in exported.graph.input[0].type.tensor_type.shape.dim]
    output_dims = [dim.dim_value for dim in exported.graph.output[0].type.tensor_type.shape.dim]
    assert input_dims == [1, NUM_CHANNELS, SEQUENCE_LENGTH]
    assert output_dims == [1, NUM_CLASSES]


def test_exported_fencenet_onnx_matches_torch_logits(tmp_path: Path):
    torch.manual_seed(11)
    model = build_model(None, allow_random=True)
    output = export_model(model, tmp_path / "fencenet_v2.onnx")
    max_abs_diff = check_onnx_against_torch(model, output, atol=1e-4)
    assert max_abs_diff is not None
    assert max_abs_diff <= 1e-4
