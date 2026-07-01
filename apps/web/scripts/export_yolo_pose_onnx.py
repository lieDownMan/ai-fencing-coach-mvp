"""Export YOLOv8n pose to the Android app asset expected by YoloPoseBackend."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("YOLO_CONFIG_DIR", str(REPO_ROOT / ".ultralytics"))

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")


DEFAULT_MODEL = "yolov8n-pose.pt"
DEFAULT_OUTPUT = REPO_ROOT / "android/app/src/main/assets/yolo_pose.onnx"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export YOLO pose to ONNX for Android.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ultralytics pose model path/name.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--opset", type=int, default=18)
    return parser.parse_args()


def main() -> None:
    from ultralytics import YOLO
    import onnxruntime as ort

    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)
    exported = Path(
        model.export(
            format="onnx",
            imgsz=args.imgsz,
            opset=args.opset,
            simplify=False,
            dynamic=False,
            nms=False,
        )
    )
    if exported.resolve() != args.output.resolve():
        args.output.unlink(missing_ok=True)
        exported.replace(args.output)

    session = ort.InferenceSession(str(args.output), providers=["CPUExecutionProvider"])
    inputs = [(item.name, item.shape, item.type) for item in session.get_inputs()]
    outputs = [(item.name, item.shape, item.type) for item in session.get_outputs()]
    print(f"Exported {args.output}")
    print(f"Inputs: {inputs}")
    print(f"Outputs: {outputs}")


if __name__ == "__main__":
    main()
