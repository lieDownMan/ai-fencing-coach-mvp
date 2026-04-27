"""
Sliding Window Inference — Action Spotter for continuous videos.

Slides a fixed-size window across a normalized skeleton timeline,
classifies each window with FenceNetV2, then applies Non-Maximum
Suppression (NMS) to merge overlapping detections.

Spec reference: fixing_app.md § Module 1
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from ..models.fencenet_v2 import FenceNetV2
from ..data.fencing_dataset import CLASS_NAMES, NUM_CHANNELS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults (aligned with trained model and spec)
# ---------------------------------------------------------------------------
WINDOW_SIZE = 28          # must match trained model input time-steps
STRIDE = 10              # ~0.33 s at 30 fps (spec value)
CONFIDENCE_THRESHOLD = 0.6
ATTACKING_ACTIONS = {"R", "IS", "WW", "JS"}


class SlidingWindowInference:
    """
    Slide a fixed-size window over a skeleton timeline, classify each window
    with FenceNetV2, and apply NMS to produce a clean action timeline.

    Parameters
    ----------
    model_path : str or Path, optional
        Path to ``best_model.pth``.  If *None* the model runs with random
        weights (useful for smoke-testing the pipeline).
    device : str
        ``"cuda"`` or ``"cpu"``.
    window_size : int
        Number of frames per window (must match model training).
    stride : int
        Step size between consecutive windows.
    confidence_threshold : float
        Windows below this softmax confidence are labelled ``Idle``.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cpu",
        window_size: int = WINDOW_SIZE,
        stride: int = STRIDE,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
    ):
        self.device = torch.device(device)
        self.window_size = window_size
        self.stride = stride
        self.confidence_threshold = confidence_threshold
        self.class_names = list(CLASS_NAMES)

        # Build model
        self.model = FenceNetV2(input_channels=NUM_CHANNELS)
        if model_path is not None:
            self._load_weights(model_path)
        self.model.to(self.device)
        self.model.eval()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        skeleton_array: np.ndarray,
    ) -> List[Dict[str, Any]]:
        """
        Run sliding-window inference + NMS on a full skeleton timeline.

        Parameters
        ----------
        skeleton_array : np.ndarray
            Shape ``(T, 9, 2)`` — spatially normalized skeleton sequence.

        Returns
        -------
        list[dict]
            Merged action segments, each containing::

                {"start_frame": int, "end_frame": int,
                 "action": str, "confidence": float}
        """
        raw_windows = self._classify_windows(skeleton_array)
        merged = self._nms(raw_windows)
        return merged

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _classify_windows(
        self, skeleton_array: np.ndarray
    ) -> List[Dict[str, Any]]:
        """Produce one prediction per sliding window."""
        T = skeleton_array.shape[0]
        results: List[Dict[str, Any]] = []

        if T < self.window_size:
            logger.warning(
                "Video has %d frames, less than window_size=%d — skipping.",
                T,
                self.window_size,
            )
            return results

        # Build all windows
        starts = list(range(0, T - self.window_size + 1, self.stride))
        if not starts:
            return results

        windows = np.stack(
            [skeleton_array[s : s + self.window_size] for s in starts],
            axis=0,
        )  # (N, window_size, 9, 2)

        # Reshape: (N, ws, 9, 2) → (N, ws, 18) → (N, 18, ws)
        N = windows.shape[0]
        flat = windows.reshape(N, self.window_size, -1)      # (N, ws, 18)
        tensor = (
            torch.from_numpy(flat)
            .float()
            .permute(0, 2, 1)                                 # (N, 18, ws)
            .to(self.device)
        )

        # Batch inference
        with torch.no_grad():
            logits = self.model(tensor)                       # (N, 6)
            probs = F.softmax(logits, dim=1)                  # (N, 6)
            confs, preds = probs.max(dim=1)                   # (N,), (N,)

        confs = confs.cpu().numpy()
        preds = preds.cpu().numpy()

        for i, start in enumerate(starts):
            pred_idx = int(preds[i])
            conf = float(confs[i])
            action = self.class_names[pred_idx] if conf >= self.confidence_threshold else "Idle"
            results.append(
                {
                    "start_frame": start,
                    "end_frame": start + self.window_size,
                    "action": action,
                    "confidence": conf,
                    "class_idx": pred_idx,
                }
            )

        return results

    def _nms(self, windows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Non-Maximum Suppression: merge consecutive overlapping windows
        with the same *attacking* action, keeping only the highest
        confidence occurrence.

        Non-attacking actions (SF, SB) and Idle are kept individually
        (but Idle is filtered out of final output).
        """
        if not windows:
            return []

        merged: List[Dict[str, Any]] = []
        current_group: List[Dict[str, Any]] = [windows[0]]

        for win in windows[1:]:
            prev = current_group[-1]
            same_action = win["action"] == prev["action"]
            is_attacking = win["action"] in ATTACKING_ACTIONS
            overlapping = win["start_frame"] < prev["end_frame"]

            if same_action and is_attacking and overlapping:
                # Extend the current group
                current_group.append(win)
            else:
                # Flush current group
                merged.append(self._best_of_group(current_group))
                current_group = [win]

        # Flush last group
        merged.append(self._best_of_group(current_group))

        # Remove Idle segments from final output
        return [seg for seg in merged if seg["action"] != "Idle"]

    @staticmethod
    def _best_of_group(group: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Return the window with the highest confidence, spanning the full group range."""
        best = max(group, key=lambda w: w["confidence"])
        return {
            "start_frame": group[0]["start_frame"],
            "end_frame": group[-1]["end_frame"],
            "action": best["action"],
            "confidence": best["confidence"],
        }

    def _load_weights(self, model_path: str) -> None:
        """Load trained weights from ``best_model.pth``."""
        path = Path(model_path)
        if not path.exists():
            logger.warning("Model weights not found: %s — using random weights.", path)
            return

        checkpoint = torch.load(str(path), map_location=self.device, weights_only=True)

        # Support both plain state_dict and wrapped checkpoint formats
        if isinstance(checkpoint, dict):
            if "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            elif "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            elif self._looks_like_state_dict(checkpoint):
                state_dict = checkpoint
            else:
                logger.warning("Unrecognized checkpoint format — using random weights.")
                return
        else:
            logger.warning("Checkpoint is not a dict — using random weights.")
            return

        self.model.load_state_dict(state_dict)
        logger.info("Loaded FenceNetV2 weights from %s", path)

    @staticmethod
    def _looks_like_state_dict(d: dict) -> bool:
        return bool(d) and all(hasattr(v, "shape") for v in d.values())
