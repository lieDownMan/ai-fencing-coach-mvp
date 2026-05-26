"""
Export FenceNetV2 best_model.pth → TFLite (float32) for on-device inference.

Usage:
    cd /path/to/ai-fencing-coach-mvp/backend
    PYTHONPATH=. ../venv/bin/python scripts/export_tflite.py

Output:
    frontend/assets/models/fencenet_v2.tflite
    (input:  [1, 18, 28]  float32)
    (output: [1,  6]      float32 logits)
"""

import sys
import os
from pathlib import Path

# ── locate project root ──────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
REPO_ROOT   = BACKEND_DIR.parent

sys.path.insert(0, str(BACKEND_DIR))

import torch
from src.models.fencenet_v2 import FenceNetV2

# ── paths ────────────────────────────────────────────────────────────────────
CHECKPOINT   = BACKEND_DIR / "weights" / "fencenet" / "best_model.pth"
OUT_DIR      = REPO_ROOT / "frontend" / "assets" / "models"
ONNX_PATH    = OUT_DIR / "fencenet_v2.onnx"
TFLITE_PATH  = OUT_DIR / "fencenet_v2.tflite"

OUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"[1/4] Loading checkpoint: {CHECKPOINT}")
model = FenceNetV2(input_channels=18)
checkpoint = torch.load(str(CHECKPOINT), map_location="cpu", weights_only=True)
if isinstance(checkpoint, dict):
    if "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    elif "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
else:
    raise RuntimeError("Unexpected checkpoint format")

model.eval()
print("    ✓ Weights loaded")

# ── Export to ONNX ───────────────────────────────────────────────────────────
print(f"[2/4] Exporting to ONNX: {ONNX_PATH}")
dummy_input = torch.zeros(1, 18, 28)  # (batch=1, channels=18, time=28)

torch.onnx.export(
    model,
    dummy_input,
    str(ONNX_PATH),
    opset_version=12,
    input_names=["input"],
    output_names=["logits"],
    dynamic_axes={
        "input":  {0: "batch"},
        "logits": {0: "batch"},
    },
)
print("    ✓ ONNX exported")

# ── ONNX → TFLite ────────────────────────────────────────────────────────────
print(f"[3/4] Converting ONNX → TFLite: {TFLITE_PATH}")
try:
    import onnx
    from onnx_tf.backend import prepare
    import tensorflow as tf

    onnx_model = onnx.load(str(ONNX_PATH))
    tf_rep = prepare(onnx_model)

    # Save as TF SavedModel first
    saved_model_dir = str(OUT_DIR / "fencenet_saved_model")
    tf_rep.export_graph(saved_model_dir)

    # Convert SavedModel → TFLite
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    tflite_model = converter.convert()

    with open(str(TFLITE_PATH), "wb") as f:
        f.write(tflite_model)
    print("    ✓ TFLite exported via onnx-tf")

except ImportError:
    print("    onnx-tf not found, trying onnx2tf ...")
    try:
        import subprocess
        result = subprocess.run(
            [
                sys.executable, "-m", "onnx2tf",
                "-i", str(ONNX_PATH),
                "-o", str(OUT_DIR / "onnx2tf_out"),
                "--non_verbose",
            ],
            capture_output=True, text=True, check=True
        )
        print(result.stdout)
        # Locate the generated tflite
        import glob
        candidates = glob.glob(str(OUT_DIR / "onnx2tf_out" / "*.tflite"))
        if candidates:
            import shutil
            shutil.copy(candidates[0], str(TFLITE_PATH))
            print(f"    ✓ TFLite exported via onnx2tf → {TFLITE_PATH}")
        else:
            raise FileNotFoundError("onnx2tf did not produce a .tflite file")
    except Exception as e2:
        print(f"\n[ERROR] Both onnx-tf and onnx2tf failed: {e2}")
        print(
            "\nPlease install one of:\n"
            "  pip install onnx onnx-tf tensorflow\n"
            "  pip install onnx onnx2tf\n"
            "Then re-run this script."
        )
        sys.exit(1)

# ── Quick sanity check ───────────────────────────────────────────────────────
print("[4/4] Sanity check ...")
try:
    import tensorflow as tf
    interp = tf.lite.Interpreter(model_path=str(TFLITE_PATH))
    interp.allocate_tensors()
    inp = interp.get_input_details()
    out = interp.get_output_details()
    print(f"    Input:  {inp[0]['shape']}  dtype={inp[0]['dtype']}")
    print(f"    Output: {out[0]['shape']}  dtype={out[0]['dtype']}")
    print("    ✓ TFLite model is valid")
except Exception as e:
    print(f"    (Sanity check skipped: {e})")

print(f"\n✅  Done!  TFLite model saved to:\n    {TFLITE_PATH}")
print(f"\nFile size: {TFLITE_PATH.stat().st_size / 1024:.1f} KB")
