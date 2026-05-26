"""Export the trained FenceNetV2 checkpoint for Android ONNX Runtime.

The Android app expects:

- input name: ``skeleton_window``
- input shape: float32[1, 18, 28]
- output name: ``logits``
- output shape: float32[1, 6]
- class order: R, IS, WW, JS, SF, SB
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from src.data.fencing_dataset import CLASS_NAMES, NUM_CHANNELS, SEQUENCE_LENGTH
from src.models.fencenet_v2 import FenceNetV2


DEFAULT_CHECKPOINT = Path("weights/fencenet/best_model.pth")
DEFAULT_OUTPUT = Path("android/app/src/main/assets/fencenet_v2.onnx")
INPUT_NAME = "skeleton_window"
OUTPUT_NAME = "logits"


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(str(path), map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(str(path), map_location="cpu")


def _extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            candidate = checkpoint.get(key)
            if isinstance(candidate, dict):
                return candidate
        if checkpoint and all(torch.is_tensor(value) for value in checkpoint.values()):
            return checkpoint
    raise ValueError("Unsupported FenceNet checkpoint format")


def build_model(
    checkpoint_path: Path | None = DEFAULT_CHECKPOINT,
    *,
    allow_random: bool = False,
) -> FenceNetV2:
    model = FenceNetV2(input_channels=NUM_CHANNELS)
    if checkpoint_path is not None and checkpoint_path.exists():
        checkpoint = _torch_load(checkpoint_path)
        state_dict = _extract_state_dict(checkpoint)
        model.load_state_dict(state_dict, strict=True)
    elif not allow_random:
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. "
            "Pass --allow-random only for tooling smoke tests."
        )
    model.eval()
    return model


def export_model(
    model: FenceNetV2,
    output_path: Path = DEFAULT_OUTPUT,
    *,
    opset_version: int = 17,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sample = torch.randn(1, NUM_CHANNELS, SEQUENCE_LENGTH, dtype=torch.float32)
    with torch.no_grad():
        torch.onnx.export(
            model,
            sample,
            str(output_path),
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=[INPUT_NAME],
            output_names=[OUTPUT_NAME],
            dynamic_axes=None,
        )
    return output_path


def check_onnx_against_torch(
    model: FenceNetV2,
    output_path: Path,
    *,
    atol: float = 1e-4,
) -> float | None:
    try:
        import numpy as np
        import onnxruntime as ort
    except ImportError:
        return None

    sample = torch.randn(1, NUM_CHANNELS, SEQUENCE_LENGTH, dtype=torch.float32)
    with torch.no_grad():
        torch_logits = model(sample).detach().cpu().numpy()

    session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    onnx_logits = session.run([OUTPUT_NAME], {INPUT_NAME: sample.numpy()})[0]
    max_abs_diff = float(np.max(np.abs(torch_logits - onnx_logits)))
    if max_abs_diff > atol:
        raise AssertionError(
            f"ONNX logits differ from PyTorch by {max_abs_diff:.6f}; "
            f"expected <= {atol}"
        )
    return max_abs_diff


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export FenceNetV2 to ONNX for Android.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument(
        "--allow-random",
        action="store_true",
        help="Export random weights if the checkpoint is absent. Use only for smoke tests.",
    )
    parser.add_argument(
        "--skip-check",
        action="store_true",
        help="Skip ONNX Runtime parity check after export.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = build_model(args.checkpoint, allow_random=args.allow_random)
    output_path = export_model(model, args.output, opset_version=args.opset)
    print(f"Exported {output_path}")
    print(f"Class order: {', '.join(CLASS_NAMES)}")
    if not args.skip_check:
        diff = check_onnx_against_torch(model, output_path)
        if diff is None:
            print("Skipped parity check because onnxruntime is not installed.")
        else:
            print(f"ONNX parity max_abs_diff={diff:.6f}")


if __name__ == "__main__":
    main()
