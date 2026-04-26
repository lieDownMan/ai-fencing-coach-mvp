#!/usr/bin/env python3
"""
train_fencenet.py — Full training pipeline for FenceNet v2.

Features:
  • Grouped K-Fold cross-validation (Prevents Data Leakage)
  • AdamW + CosineAnnealingLR for stable convergence
  • Label Smoothing for better generalization
  • Majority-voting evaluation per video
  • Per-epoch confusion matrix logging
  • Best-model checkpointing → weights/fencenet/best_model.pth

Usage:
    python train_fencenet.py --data_dir data/json_samples
"""

import argparse
import copy
import json
import logging
import os
import sys
import random
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Imports from project
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data.fencing_dataset import (
    CLASS_NAMES,
    NUM_CHANNELS,
    NUM_CLASSES,
    FencingDataset,
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

def build_grouped_folds(
    dataset: FencingDataset,
    n_folds: int = 5,
    seed: int = 42,
) -> List[Tuple[List[int], List[int]]]:
    """
    Build Grouped K-fold splits to prevent Data Leakage.
    Ensures all augmented versions of the same base video stay in the same fold.
    """
    groups = defaultdict(list)
    
    for i, sample in enumerate(dataset.samples):
        # 嘗試從 sample 中取得 file_path，若無則依賴順序推斷 (假設1影片=6增強版本)
        file_path = sample.get("file_path", "")
        if file_path:
            base_name = os.path.basename(file_path).split("_orig")[0].split("_flip")[0].split("_noise")[0].split("_twarp")[0]
        else:
            base_name = f"dummy_video_{i // 6}"
            
        groups[base_name].append(i)

    unique_videos = list(groups.keys())
    rng = random.Random(seed)
    rng.shuffle(unique_videos)

    folds = []
    chunk_size = max(1, len(unique_videos) // n_folds)
    
    for fold_id in range(n_folds):
        start_idx = fold_id * chunk_size
        end_idx = (fold_id + 1) * chunk_size if fold_id < n_folds - 1 else len(unique_videos)
        
        val_videos = set(unique_videos[start_idx:end_idx])
            
        train_idx, val_idx = [], []
        for vid, indices in groups.items():
            if vid in val_videos:
                val_idx.extend(indices)
            else:
                train_idx.extend(indices)
                
        folds.append((train_idx, val_idx))
        logger.info(f"  Fold {fold_id + 1:>2d}: train={len(train_idx)} val={len(val_idx)}")

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
) -> Tuple[float, np.ndarray]:
    model.eval()

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

        subseqs = subseqs.to(device)
        logits = model(subseqs)                # (N, 6)
        preds = logits.argmax(dim=1).cpu().numpy()

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

    # 針對 Underfitting：降低 Dropout 讓模型保有更多學習能力
    model = FenceNetV2(
        input_channels=NUM_CHANNELS,
        kernel_size=3,
        dropout=0.1, 
    ).to(device)

    # 針對 Underfitting/Overconfidence：加入 label_smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # 使用 AdamW 取代 Adam，幫助模型更穩定收斂
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # 餘弦退火排程器，主動調節學習率
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
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
            features = features.to(device)     
            labels = labels.to(device)         

            optimizer.zero_grad()
            logits = model(features)           
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1)
            running_correct += (preds == labels).sum().item()
            running_total += labels.size(0)

        train_loss = running_loss / running_total
        train_acc = running_correct / running_total * 100

        # ---------- Validation ----------
        val_acc, cm = evaluate_majority_voting(model, dataset, val_indices, device)
        
        # 餘弦退火是在每個 epoch 結束時強制 step
        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        logger.info(
            f"Fold {fold_idx} │ Epoch {epoch:>3d}/{args.epochs} │ "
            f"Train Loss {train_loss:.4f}  Acc {train_acc:5.1f}% │ "
            f"Val Acc {val_acc:5.1f}% │ LR {current_lr:.2e}"
        )

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
    parser = argparse.ArgumentParser(description="Train FenceNet v2 with Grouped K-fold CV.")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20) # 放寬耐心值，讓模型有時間爬出局部解
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--output_dir", type=str, default="weights/fencenet")
    parser.add_argument("--log_dir", type=str, default="logs/fencenet")
    args = parser.parse_args()

    # --- 設定 log 同時輸出到檔案 ---
    os.makedirs(args.log_dir, exist_ok=True)
    log_file_path = os.path.join(args.log_dir, "train.log")
    file_handler = logging.FileHandler(log_file_path, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s │ %(levelname)-7s │ %(message)s",
                          datefmt="%Y-%m-%d %H:%M:%S")
    )
    logger.addHandler(file_handler)
    logger.info(f"Training log will be saved to → {log_file_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    full_dataset = FencingDataset(data_dir=args.data_dir, is_train=True)
    logger.info(f"Loaded {len(full_dataset)} samples from {args.data_dir}")

    folds = build_grouped_folds(full_dataset, n_folds=args.n_folds)
    logger.info(f"Number of folds: {len(folds)}")

    fold_results: List[dict] = []
    overall_best_acc = 0.0
    overall_best_state = None

    for fold_idx, (train_idx, val_idx) in enumerate(folds, 1):
        logger.info(f"\n{'='*60}\n  FOLD {fold_idx} / {len(folds)}\n{'='*60}")

        acc, state = train_one_fold(fold_idx, full_dataset, train_idx, val_idx, device, args)
        fold_results.append({"fold": fold_idx, "val_accuracy": acc})

        if acc > overall_best_acc:
            overall_best_acc = acc
            overall_best_state = state

    accs = [r["val_accuracy"] for r in fold_results]
    mean_acc = np.mean(accs)
    std_acc = np.std(accs)
    logger.info(f"\n{'='*60}\n  CROSS-VALIDATION RESULTS\n{'='*60}")
    for r in fold_results:
        logger.info(f"  Fold {r['fold']:>2d}: {r['val_accuracy']:5.1f}%")
    logger.info(f"  Mean accuracy: {mean_acc:.1f}% ± {std_acc:.1f}%")
    logger.info(f"  Best single-fold accuracy: {overall_best_acc:.1f}%")

    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, "best_model.pth")
    if overall_best_state is not None:
        torch.save(overall_best_state, save_path)
        logger.info(f"  Best model saved → {save_path}")

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