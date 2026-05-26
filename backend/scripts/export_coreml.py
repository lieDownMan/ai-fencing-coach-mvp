"""
Export FenceNetV2 best_model.pth → CoreML (.mlpackage) for iOS on-device inference.
CoreML is native on iOS and works perfectly with Flutter via MethodChannel.

Usage:
    cd /path/to/ai-fencing-coach-mvp
    PYTHONPATH=backend venv/bin/python backend/scripts/export_coreml.py

Output:
    frontend/assets/models/fencenet_v2.mlpackage  (for iOS embedding)
    frontend/ios/Runner/fencenet_v2.mlpackage      (for Xcode linking)
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = REPO_ROOT / "backend"
sys.path.insert(0, str(BACKEND_DIR))

import torch
import torch.nn as nn
import coremltools as ct
from src.models.fencenet_v2 import FenceNetV2

CHECKPOINT = BACKEND_DIR / "weights" / "fencenet" / "best_model.pth"
OUT_ASSET  = REPO_ROOT / "frontend" / "assets" / "models" / "fencenet_v2.mlpackage"
OUT_IOS    = REPO_ROOT / "frontend" / "ios" / "Runner" / "fencenet_v2.mlpackage"

# ── Load model ───────────────────────────────────────────────────────────────
print(f"[1/4] Loading checkpoint: {CHECKPOINT}")
model = FenceNetV2(input_channels=18)
ckpt = torch.load(str(CHECKPOINT), map_location="cpu", weights_only=True)
if isinstance(ckpt, dict):
    sd = ckpt.get("state_dict") or ckpt.get("model_state_dict") or ckpt
    model.load_state_dict(sd)
model.eval()
print("    ✓ Weights loaded")

# ── Trace the model ───────────────────────────────────────────────────────────
print("[2/4] Tracing model...")
dummy = torch.zeros(1, 18, 28)
with torch.no_grad():
    traced = torch.jit.trace(model, dummy)
print("    ✓ Model traced")

# ── Convert to CoreML ────────────────────────────────────────────────────────
print("[3/4] Converting to CoreML...")

input_spec = ct.TensorType(
    name="input",
    shape=(1, 18, 28),
    dtype=float,
)

mlmodel = ct.convert(
    traced,
    inputs=[input_spec],
    outputs=[ct.TensorType(name="logits")],
    minimum_deployment_target=ct.target.iOS16,
    compute_units=ct.ComputeUnit.ALL,
)

# Add metadata
mlmodel.short_description = "FenceNetV2 fencing action classifier"
mlmodel.input_description["input"] = "(1, 18, 28) — 9 joints × 2 coords × 28 frames"
mlmodel.output_description["logits"] = "Logits for [R, IS, WW, JS, SF, SB]"
print("    ✓ Converted to CoreML")

# ── Save ─────────────────────────────────────────────────────────────────────
print(f"[4/4] Saving...")
OUT_ASSET.parent.mkdir(parents=True, exist_ok=True)
mlmodel.save(str(OUT_ASSET))
print(f"    ✓ Saved to {OUT_ASSET}")

# Also copy to iOS runner dir for direct Xcode linking
import shutil
OUT_IOS.parent.mkdir(parents=True, exist_ok=True)
if OUT_IOS.exists():
    shutil.rmtree(str(OUT_IOS))
shutil.copytree(str(OUT_ASSET), str(OUT_IOS))
print(f"    ✓ Copied to {OUT_IOS}")

size_mb = sum(f.stat().st_size for f in OUT_ASSET.rglob('*') if f.is_file()) / 1e6
print(f"\n✅ Done!  CoreML model size: {size_mb:.1f} MB")
print(f"   Classes: ['R', 'IS', 'WW', 'JS', 'SF', 'SB']")
print(f"   Input:  [1, 18, 28]")
print(f"   Output: [1, 6]")
