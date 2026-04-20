#!/usr/bin/env python3
"""Prepare the public FFD dataset into model-ready FenceNet windows."""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.training import prepare_ffd_dataset, save_prepared_dataset


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare the Fencing Footwork Dataset (FFD) for FenceNet training."
    )
    parser.add_argument(
        "--ffd-root",
        required=True,
        help="Path to the unpacked FFD dataset root containing *_Body.mat files.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to write the prepared .npz dataset bundle.",
    )
    parser.add_argument(
        "--summary-json",
        type=str,
        help="Optional path to write dataset summary metadata as JSON.",
    )
    parser.add_argument(
        "--windows-per-sequence",
        type=int,
        default=10,
        help="Maximum number of 28-frame windows to sample per source sequence.",
    )
    parser.add_argument(
        "--max-random-start",
        type=int,
        default=20,
        help="Maximum starting-frame offset for paper-style random FFD windows.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for window sampling.",
    )
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    dataset = prepare_ffd_dataset(
        dataset_root=Path(args.ffd_root),
        windows_per_sequence=args.windows_per_sequence,
        max_random_start=args.max_random_start,
        random_seed=args.seed,
    )
    output_path = Path(args.output).expanduser()
    save_prepared_dataset(dataset, output_path)

    summary = dataset.summary()
    print(f"Prepared dataset written: {output_path}")
    print(json.dumps(summary, indent=2, sort_keys=True))

    if args.summary_json:
        summary_path = Path(args.summary_json).expanduser()
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(
                {
                    **summary,
                    "source_root": str(Path(args.ffd_root).expanduser()),
                    "output_path": str(output_path),
                    "windows_per_sequence": args.windows_per_sequence,
                    "max_random_start": args.max_random_start,
                    "seed": args.seed,
                },
                indent=2,
                sort_keys=True,
            ) + "\n",
            encoding="utf-8",
        )
        print(f"Summary JSON written: {summary_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
