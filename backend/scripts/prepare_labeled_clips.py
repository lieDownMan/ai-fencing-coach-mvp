#!/usr/bin/env python3
"""Prepare a custom labeled-clip CSV into model-ready FenceNet samples."""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.training import (
    prepare_labeled_video_dataset,
    save_prepared_dataset,
    write_clip_label_template,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare custom labeled video clips for FenceNet training."
    )
    parser.add_argument(
        "--labels-csv",
        type=str,
        help="CSV with video_path,label,start_frame,end_frame,... columns.",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to write the prepared .npz dataset bundle.",
    )
    parser.add_argument(
        "--pose-backend",
        choices=["auto", "ultralytics", "mock"],
        default="ultralytics",
        help="Pose backend used to extract skeletons from the labeled clips.",
    )
    parser.add_argument(
        "--pose-model",
        type=str,
        default=None,
        help="Optional Ultralytics pose model path.",
    )
    parser.add_argument(
        "--summary-json",
        type=str,
        help="Optional path to write dataset summary metadata as JSON.",
    )
    parser.add_argument(
        "--write-template",
        type=str,
        help="Write a starter clip-label CSV template and exit.",
    )
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.write_template:
        template_path = Path(args.write_template).expanduser()
        write_clip_label_template(template_path)
        print(f"Clip-label CSV template written: {template_path}")
        return 0

    if not args.labels_csv or not args.output:
        parser.error("--labels-csv and --output are required unless --write-template is used")

    dataset = prepare_labeled_video_dataset(
        csv_path=Path(args.labels_csv),
        pose_backend=args.pose_backend,
        pose_model_path=args.pose_model,
    )
    output_path = Path(args.output).expanduser()
    save_prepared_dataset(dataset, output_path)

    summary = dataset.summary()
    print(f"Prepared labeled dataset written: {output_path}")
    print(json.dumps(summary, indent=2, sort_keys=True))

    if args.summary_json:
        summary_path = Path(args.summary_json).expanduser()
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(
                {
                    **summary,
                    "labels_csv": str(Path(args.labels_csv).expanduser()),
                    "output_path": str(output_path),
                    "pose_backend": args.pose_backend,
                    "pose_model": args.pose_model,
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
