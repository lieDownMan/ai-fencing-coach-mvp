#!/usr/bin/env python3
"""
train_fencenet.py — Full training pipeline for FenceNet v2.

Features:
  • Leave-One-Person-Out (LOPO) 10-fold cross-validation
  • ReduceLROnPlateau scheduler
  • Early stopping
  • Majority-voting evaluation per video
  • Per-epoch confusion matrix logging
  • Best-model checkpointing → weights/fencenet/best_model.pth

Usage:
    python train_fencenet.py --data_dir <path_to_json_samples>

See FENCENET_TRAINING.md for the full specification.
"""

import argparse
import copy
import json
import logging
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

# ---------------------------------------------------------------------------
# Imports from project
# ---------------------------------------------------------------------------

# Allow running from project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data.fencing_dataset import (
    CLASS_NAMES,
    NUM_CHANNELS,
    NUM_CLASSES,
    FencingDataset,
    eval_collate_fn,
)
from src.models.fencenet_v2 import FenceNetV2

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train_fencenet")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def build_lopo_folds(
    dataset: FencingDataset,
) -> List[Tuple[List[int], List[int]]]:
    """
    Build Leave-One-Person-Out folds.

    Returns a list of (train_indices, val_indices) tuples, one per unique
    fencer.
    """
    fencer_ids = dataset.get_fencer_ids()
    unique_ids = dataset.get_unique_fencer_ids()

    folds = []
    for held_out in unique_ids:
        train_idx = [i for i, fid in enumerate(fencer_ids) if fid != held_out]
        val_idx = [i for i, fid in enumerate(fencer_ids) if fid == held_out]
        folds.append((train_idx, val_idx))
        logger.info(
            f"  Fold {len(folds):>2d}: hold-out={held_out!r}  "
            f"train={len(train_idx)}  val={len(val_idx)}"
        )

    return folds


def confusion_matrix_str(cm: np.ndarray, class_names: List[str]) -> str:
    """Pretty-print a confusion matrix."""
    header = "       " + "  ".join(f"{n:>4s}" for n in class_names)
    lines = [header]
    for i, row in enumerate(cm):
        row_str = "  ".join(f"{int(v):>4d}" for v in row)
        lines.append(f"  {class_names[i]:>4s}  {row_str}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Majority-voting evaluation
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate_majority_voting(
    model: nn.Module,
    dataset: FencingDataset,
    indices: List[int],
    device: torch.device,
    batch_size: int = 64,
) -> Tuple[float, np.ndarray]:
    """
    Evaluate with majority voting — each video may yield multiple 28-frame
    subsequences; the predicted label is the mode of sub-predictions.

    Returns:
        accuracy:  float, percentage of correctly classified videos.
        cm:        np.ndarray of shape (NUM_CLASSES, NUM_CLASSES), confusion
                   matrix where cm[true][pred] is the count.
    """
    model.eval()

    # Build evaluation dataset (is_train=False → returns all subsequences)
    eval_ds = FencingDataset(
        data_dir=dataset.data_dir,
        sample_indices=indices,
        is_train=False,
    )

    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    correct = 0
    total = 0

    for i in range(len(eval_ds)):
        subseqs, label = eval_ds[i]           # subseqs: (N, 18, 28)
        true_label = label.item()

        # Run all subsequences through the model
        subseqs = subseqs.to(device)
        logits = model(subseqs)                # (N, 6)
        preds = logits.argmax(dim=1).cpu().numpy()

        # Majority vote
        counter = Counter(preds.tolist())
        pred_label = counter.most_common(1)[0][0]

        cm[true_label][pred_label] += 1
        if pred_label == true_label:
            correct += 1
        total += 1

    accuracy = correct / total * 100 if total > 0 else 0.0
    return accuracy, cm


# ---------------------------------------------------------------------------
# Single-fold training
# ---------------------------------------------------------------------------


def train_one_fold(
    fold_idx: int,
    dataset: FencingDataset,
    train_indices: List[int],
    val_indices: List[int],
    device: torch.device,
    args: argparse.Namespace,
) -> Tuple[float, nn.Module]:
    """
    Train FenceNetV2 for one fold.

    Returns (best_val_accuracy, best_model_state_dict).
    """

    # --- Data loaders (training uses random crop → is_train=True) ---
    train_ds = FencingDataset(
        data_dir=args.data_dir,
        sample_indices=train_indices,
        is_train=True,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # --- Model ---
    model = FenceNetV2(
        input_channels=NUM_CHANNELS,
        kernel_size=3,
        dropout=0.2,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=8, verbose=True,
    )

    best_acc = 0.0
    best_state = None
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        # ---------- Training ----------
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for features, labels in train_loader:
            features = features.to(device)     # (B, 18, 28)
            labels = labels.to(device)         # (B,)

            optimizer.zero_grad()
            logits = model(features)           # (B, 6)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1)
            running_correct += (preds == labels).sum().item()
            running_total += labels.size(0)

        train_loss = running_loss / running_total
        train_acc = running_correct / running_total * 100

        # ---------- Validation (majority voting) ----------
        val_acc, cm = evaluate_majority_voting(
            model, dataset, val_indices, device, batch_size=args.batch_size,
        )
        scheduler.step(val_acc)

        current_lr = optimizer.param_groups[0]["lr"]
        logger.info(
            f"Fold {fold_idx} │ Epoch {epoch:>3d}/{args.epochs} │ "
            f"Train Loss {train_loss:.4f}  Acc {train_acc:5.1f}% │ "
            f"Val Acc {val_acc:5.1f}% │ LR {current_lr:.2e}"
        )

        # Log confusion matrix every 10 epochs or at the last epoch
        if epoch % 10 == 0 or epoch == args.epochs:
            logger.info(
                f"Fold {fold_idx} │ Confusion Matrix (epoch {epoch}):\n"
                + confusion_matrix_str(cm, CLASS_NAMES)
            )

        # ---------- Check best & early stopping ----------
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            logger.info(f"  ★ New best val accuracy: {best_acc:.1f}%")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logger.info(
                    f"  Early stopping at epoch {epoch} "
                    f"(no improvement for {args.patience} epochs)"
                )
                break

    return best_acc, best_state


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Train FenceNet v2 with LOPO cross-validation."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing JSON sample files.",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15,
                        help="Early stopping patience (epochs).")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="weights/fencenet",
        help="Where to save the best model weights.",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="logs/fencenet",
        help="Directory for training logs (JSON summaries).",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # --- Load full dataset (metadata only) ---
    full_dataset = FencingDataset(data_dir=args.data_dir, is_train=True)
    logger.info(f"Loaded {len(full_dataset)} samples from {args.data_dir}")
    logger.info(
        f"Unique fencers: {full_dataset.get_unique_fencer_ids()} "
        f"({len(full_dataset.get_unique_fencer_ids())} total)"
    )

    # --- Build LOPO folds ---
    folds = build_lopo_folds(full_dataset)
    logger.info(f"Number of folds: {len(folds)}")

    # --- Train each fold ---
    fold_results: List[dict] = []
    overall_best_acc = 0.0
    overall_best_state = None

    for fold_idx, (train_idx, val_idx) in enumerate(folds, 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"  FOLD {fold_idx} / {len(folds)}")
        logger.info(f"{'='*60}")

        acc, state = train_one_fold(
            fold_idx, full_dataset, train_idx, val_idx, device, args
        )
        fold_results.append({"fold": fold_idx, "val_accuracy": acc})

        if acc > overall_best_acc:
            overall_best_acc = acc
            overall_best_state = state

    # --- Summary ---
    accs = [r["val_accuracy"] for r in fold_results]
    mean_acc = np.mean(accs)
    std_acc = np.std(accs)
    logger.info(f"\n{'='*60}")
    logger.info(f"  CROSS-VALIDATION RESULTS")
    logger.info(f"{'='*60}")
    for r in fold_results:
        logger.info(f"  Fold {r['fold']:>2d}: {r['val_accuracy']:5.1f}%")
    logger.info(f"  Mean accuracy: {mean_acc:.1f}% ± {std_acc:.1f}%")
    logger.info(f"  Best single-fold accuracy: {overall_best_acc:.1f}%")

    # --- Save best model ---
    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, "best_model.pth")
    torch.save(overall_best_state, save_path)
    logger.info(f"  Best model saved → {save_path}")

    # --- Save JSON log ---
    os.makedirs(args.log_dir, exist_ok=True)
    log_path = os.path.join(args.log_dir, "cv_results.json")
    log_data = {
        "folds": fold_results,
        "mean_accuracy": float(mean_acc),
        "std_accuracy": float(std_acc),
        "best_accuracy": float(overall_best_acc),
        "args": vars(args),
    }
    with open(log_path, "w") as f:
        json.dump(log_data, f, indent=2)
    logger.info(f"  Training log saved → {log_path}")


if __name__ == "__main__":
    main()
