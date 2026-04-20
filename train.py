#!/usr/bin/env python3
"""Train FenceNet/BiFenceNet on a prepared dataset bundle."""

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.training import (
    TrainingConfig,
    build_dataloaders,
    build_model,
    load_prepared_dataset,
    split_dataset_indices,
    train_model,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train FenceNet or BiFenceNet on a prepared .npz dataset."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Prepared dataset bundle produced by scripts/prepare_ffd.py or scripts/prepare_labeled_clips.py.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where best.pt, last.pt, and metrics.json will be written.",
    )
    parser.add_argument(
        "--model-type",
        choices=["fencenet", "bifencenet"],
        default="fencenet",
        help="Model architecture to train.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Torch device for training.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Mini-batch size.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Adam learning rate.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Adam weight decay.",
    )
    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=0.2,
        help="Validation ratio when not using a holdout subject.",
    )
    parser.add_argument(
        "--holdout-subject",
        type=str,
        help="Optional subject_id to reserve entirely for validation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for random train/validation splitting.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader worker count.",
    )
    return parser


def resolve_device(value: str) -> str:
    """Resolve auto/cpu/cuda into a concrete torch device string."""
    if value == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    return value


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    dataset_path = Path(args.dataset).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)

    dataset = load_prepared_dataset(dataset_path)
    train_indices, validation_indices = split_dataset_indices(
        dataset=dataset,
        validation_ratio=args.validation_ratio,
        random_seed=args.seed,
        holdout_subject=args.holdout_subject,
    )

    train_loader, validation_loader = build_dataloaders(
        dataset=dataset,
        train_indices=train_indices,
        validation_indices=validation_indices,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    config = TrainingConfig(
        model_type=args.model_type,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
    )
    model = build_model(model_type=args.model_type, device=device)

    run_summary = train_model(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        config=config,
        checkpoint_dir=str(output_dir),
        extra_checkpoint_metadata={
            "dataset_path": str(dataset_path),
            "num_train_samples": int(len(train_indices)),
            "num_validation_samples": int(len(validation_indices)),
            "holdout_subject": args.holdout_subject,
            "validation_ratio": args.validation_ratio,
            "seed": args.seed,
        },
    )

    metrics = {
        "dataset": dataset.summary(),
        "dataset_path": str(dataset_path),
        "model_type": args.model_type,
        "device": device,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "holdout_subject": args.holdout_subject,
        "validation_ratio": args.validation_ratio,
        "seed": args.seed,
        "train_indices": train_indices.tolist(),
        "validation_indices": validation_indices.tolist(),
        **run_summary,
    }
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"Training complete. Best checkpoint: {output_dir / 'best.pt'}")
    print(f"Last checkpoint: {output_dir / 'last.pt'}")
    print(f"Metrics JSON: {metrics_path}")
    print(
        json.dumps(
            {
                "best_epoch": run_summary["best_epoch"],
                "best_validation_accuracy": run_summary["best_validation_accuracy"],
                "num_train_samples": int(len(train_indices)),
                "num_validation_samples": int(len(validation_indices)),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
