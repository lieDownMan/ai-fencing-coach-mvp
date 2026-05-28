"""Training loop utilities for FenceNet/BiFenceNet."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from ..models import BiFenceNet, FenceNet
from .checkpoints import save_training_checkpoint
from .data import ACTION_CLASSES, PreparedDataset


@dataclass
class TrainingConfig:
    """Minimal training configuration."""

    model_type: str = "fencenet"
    device: str = "cpu"
    epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    num_workers: int = 0


class PreparedSkeletonTorchDataset(Dataset):
    """Torch dataset wrapper around a prepared skeleton bundle."""

    def __init__(self, dataset: PreparedDataset, indices: Sequence[int]):
        self.samples = dataset.samples[np.asarray(indices, dtype=np.int64)]
        self.labels = dataset.labels[np.asarray(indices, dtype=np.int64)]

    def __len__(self) -> int:
        return int(self.samples.shape[0])

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = torch.from_numpy(self.samples[index]).float()
        label = torch.tensor(int(self.labels[index]), dtype=torch.long)
        return sample, label


def build_model(model_type: str, device: str) -> torch.nn.Module:
    """Construct a FenceNet/BiFenceNet model for training."""
    normalized_type = str(model_type).strip().lower()
    if normalized_type == "fencenet":
        model = FenceNet(input_channels=18, device=device)
    elif normalized_type == "bifencenet":
        model = BiFenceNet(input_channels=18, device=device)
    else:
        raise ValueError("model_type must be 'fencenet' or 'bifencenet'")
    return model.to(device)


def build_dataloaders(
    dataset: PreparedDataset,
    train_indices: Sequence[int],
    validation_indices: Sequence[int],
    batch_size: int,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader]:
    """Build train/validation dataloaders from index splits."""
    train_dataset = PreparedSkeletonTorchDataset(dataset, train_indices)
    validation_dataset = PreparedSkeletonTorchDataset(dataset, validation_indices)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, validation_loader


def train_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    config: TrainingConfig,
    checkpoint_dir: Optional[str] = None,
    extra_checkpoint_metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Train a FenceNet/BiFenceNet model and return summary metrics."""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    history: List[Dict[str, Any]] = []
    best_validation_accuracy = -1.0
    best_epoch = 0

    for epoch in range(1, config.epochs + 1):
        train_metrics = _run_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            device=config.device,
            optimizer=optimizer,
        )
        validation_metrics = evaluate_model(
            model=model,
            dataloader=validation_loader,
            criterion=criterion,
            device=config.device,
        )
        epoch_metrics = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "validation_loss": validation_metrics["loss"],
            "validation_accuracy": validation_metrics["accuracy"],
        }
        history.append(epoch_metrics)

        if checkpoint_dir:
            metadata = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "validation_loss": validation_metrics["loss"],
                "validation_accuracy": validation_metrics["accuracy"],
            }
            if extra_checkpoint_metadata:
                metadata.update(extra_checkpoint_metadata)
            save_training_checkpoint(
                output_path=f"{checkpoint_dir}/last.pt",
                model=model,
                model_type=config.model_type,
                training_metadata=metadata,
            )

        if validation_metrics["accuracy"] > best_validation_accuracy:
            best_validation_accuracy = validation_metrics["accuracy"]
            best_epoch = epoch
            if checkpoint_dir:
                metadata = {
                    "epoch": epoch,
                    "best_validation_accuracy": validation_metrics["accuracy"],
                }
                if extra_checkpoint_metadata:
                    metadata.update(extra_checkpoint_metadata)
                save_training_checkpoint(
                    output_path=f"{checkpoint_dir}/best.pt",
                    model=model,
                    model_type=config.model_type,
                    training_metadata=metadata,
                )

    return {
        "history": history,
        "best_epoch": int(best_epoch),
        "best_validation_accuracy": float(best_validation_accuracy),
        "action_classes": list(ACTION_CLASSES),
    }


def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str
) -> Dict[str, Any]:
    """Evaluate a trained model on a dataloader."""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    total_correct = 0
    confusion = np.zeros((len(ACTION_CLASSES), len(ACTION_CLASSES)), dtype=np.int64)

    with torch.no_grad():
        for samples, labels in dataloader:
            inputs = skeleton_batch_to_model_input(samples.to(device))
            labels = labels.to(device)
            logits = model(inputs)
            loss = criterion(logits, labels)

            predictions = torch.argmax(logits, dim=1)
            total_loss += float(loss.item()) * int(labels.shape[0])
            total_samples += int(labels.shape[0])
            total_correct += int((predictions == labels).sum().item())

            for true_index, predicted_index in zip(
                labels.cpu().numpy().tolist(),
                predictions.cpu().numpy().tolist(),
            ):
                confusion[int(true_index), int(predicted_index)] += 1

    average_loss = total_loss / total_samples if total_samples else 0.0
    accuracy = total_correct / total_samples if total_samples else 0.0
    return {
        "loss": float(average_loss),
        "accuracy": float(accuracy),
        "confusion_matrix": confusion.tolist(),
    }


def skeleton_batch_to_model_input(batch: torch.Tensor) -> torch.Tensor:
    """Convert (batch, time, joints, xy) tensors to FenceNet input layout."""
    if batch.ndim != 4 or batch.shape[2:] != (9, 2):
        raise ValueError("Expected batch with shape (batch, 28, 9, 2)")
    flattened = batch.reshape(batch.shape[0], batch.shape[1], -1)
    return flattened.permute(0, 2, 1)


def _run_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str,
    optimizer: Optional[torch.optim.Optimizer]
) -> Dict[str, Any]:
    """Run one train/eval epoch over a dataloader."""
    is_training = optimizer is not None
    model.train(mode=is_training)

    total_loss = 0.0
    total_samples = 0
    total_correct = 0

    for samples, labels in dataloader:
        inputs = skeleton_batch_to_model_input(samples.to(device))
        labels = labels.to(device)

        if is_training:
            optimizer.zero_grad()

        logits = model(inputs)
        loss = criterion(logits, labels)

        if is_training:
            loss.backward()
            optimizer.step()

        predictions = torch.argmax(logits, dim=1)
        total_loss += float(loss.item()) * int(labels.shape[0])
        total_samples += int(labels.shape[0])
        total_correct += int((predictions == labels).sum().item())

    average_loss = total_loss / total_samples if total_samples else 0.0
    accuracy = total_correct / total_samples if total_samples else 0.0
    return {
        "loss": float(average_loss),
        "accuracy": float(accuracy),
    }
