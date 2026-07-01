"""Checkpoint helpers for FenceNet/BiFenceNet training."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch

from .data import ACTION_CLASSES


def build_training_checkpoint(
    model: torch.nn.Module,
    model_type: str,
    training_metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Build a checkpoint payload compatible with the app loader."""
    normalized_type = str(model_type).strip().lower()
    if normalized_type not in {"fencenet", "bifencenet"}:
        raise ValueError("model_type must be 'fencenet' or 'bifencenet'")

    metadata = {
        "format_version": 1,
        "model_type": normalized_type,
        "input_channels": 18,
        "num_classes": len(ACTION_CLASSES),
        "action_classes": list(ACTION_CLASSES),
    }
    if training_metadata:
        metadata["training"] = training_metadata

    return {
        "format_version": 1,
        "model_type": normalized_type,
        "input_channels": 18,
        "num_classes": len(ACTION_CLASSES),
        "action_classes": list(ACTION_CLASSES),
        "metadata": metadata,
        "state_dict": model.state_dict(),
    }


def save_training_checkpoint(
    output_path: Path,
    model: torch.nn.Module,
    model_type: str,
    training_metadata: Optional[Dict[str, Any]] = None
):
    """Save a training checkpoint to disk."""
    output_path = Path(output_path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        build_training_checkpoint(
            model=model,
            model_type=model_type,
            training_metadata=training_metadata,
        ),
        output_path,
    )
