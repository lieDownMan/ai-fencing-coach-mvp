"""Browser entrypoint for the AI Fencing Coach MVP demo."""

import argparse
from pathlib import Path

from src.app_interface.web_demo import run_server


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI options for starting the local web demo."""
    parser = argparse.ArgumentParser(description="AI Fencing Coach browser demo")
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host interface for the web demo. Use 0.0.0.0 for remote access.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port for the web demo.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="web_outputs",
        help="Directory for generated web demo videos, reports, and profiles.",
    )
    return parser


def main() -> int:
    """Run the browser demo."""
    args = build_arg_parser().parse_args()
    run_server(
        host=args.host,
        port=args.port,
        repo_root=Path(__file__).resolve().parent,
        output_dir=Path(args.output_dir),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
