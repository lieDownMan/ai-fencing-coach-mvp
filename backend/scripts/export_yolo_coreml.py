"""
Export YOLOv8-pose model (.pt) to CoreML (.mlpackage) for iOS.
Usage:
    PYTHONPATH=. venv/bin/python backend/scripts/export_yolo_coreml.py
"""

import sys
import shutil
from pathlib import Path
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parents[2]
YOLO_PT = REPO_ROOT / "yolov8n-pose.pt"
DEST_DIR = REPO_ROOT / "frontend" / "ios" / "Runner"

def main():
    if not YOLO_PT.exists():
        print(f"Error: YOLO weights not found at {YOLO_PT}")
        sys.exit(1)

    print(f"[1/3] Loading YOLOv8-pose model from {YOLO_PT}...")
    model = YOLO(str(YOLO_PT))

    print(f"[2/3] Exporting to CoreML (nms=False, imgsz=640)...")
    # We set nms=False so the output contains the raw keypoint coordinates [1, 56, 8400]
    exported_path_str = model.export(format="coreml", nms=False, imgsz=640)
    exported_path = Path(exported_path_str)

    if not exported_path.exists():
        print(f"Error: Export failed, output not found at {exported_path}")
        sys.exit(1)

    print(f"    ✓ Exported successfully to {exported_path}")

    # Copy to iOS Runner folder
    dest_path = DEST_DIR / "yolov8n_pose.mlpackage"
    print(f"[3/3] Copying model to iOS Runner: {dest_path}...")
    
    if dest_path.exists():
        if dest_path.is_dir():
            shutil.rmtree(dest_path)
        else:
            dest_path.unlink()

    shutil.copytree(exported_path, dest_path)
    print("✅ All done!")

if __name__ == "__main__":
    main()
