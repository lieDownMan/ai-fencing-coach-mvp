"""
AI Fencing Coach & Referee System - Main Application.

This is the entry point for the fencing coaching system.
It demonstrates the complete pipeline from video input to coaching feedback.
"""

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import yaml

from src.app_interface.system_pipeline import SystemPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path("config.yaml")


def load_app_config(config_path: Optional[Path]) -> Dict[str, Any]:
    """Load optional YAML app configuration."""
    if config_path is None:
        return {}

    path = Path(config_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file) or {}

    if not isinstance(config, dict):
        raise ValueError("App config must be a YAML mapping")

    return config


def _resolve_config_path(config_arg: Optional[str]) -> Optional[Path]:
    """Resolve the CLI config argument, using config.yaml when present."""
    if config_arg:
        return Path(config_arg)
    if DEFAULT_CONFIG_PATH.exists():
        return DEFAULT_CONFIG_PATH
    return None


def _config_value(
    config: Dict[str, Any],
    *keys: str,
    default: Any = None
) -> Any:
    """Read a nested config value without assuming every section exists."""
    current: Any = config
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def _format_model_status(status: Dict[str, Any]) -> str:
    """Format model weight status for CLI output."""
    model_type = status.get("model_type", "model")
    model_weights = status.get("model_weights", "random")
    checkpoint_path = status.get("model_checkpoint")

    if model_weights == "checkpoint":
        return f"Model weights: checkpoint ({model_type}, {checkpoint_path})"
    if checkpoint_path:
        error = status.get("model_checkpoint_error") or "checkpoint was not loaded"
        return f"Model weights: random ({model_type}; {error})"
    return f"Model weights: random ({model_type}; no checkpoint provided)"


class FencingCoachApplication:
    """Main application controller."""

    def __init__(
        self,
        use_bifencenet: bool = False,
        device: str = "auto",
        model_checkpoint: Optional[Path] = None,
        pose_backend: str = "auto",
        pose_model: Optional[str] = None,
        profiles_dir: str = "data/fencer_profiles/",
        llm_model_name: str = "llava-next",
        ui_width: int = 1600,
        ui_height: int = 900,
        create_ui: bool = False
    ):
        """
        Initialize application.

        Args:
            use_bifencenet: Use BiFenceNet instead of FenceNet
            device: Device to use (auto, cuda, cpu)
            model_checkpoint: Path to pretrained model
            pose_backend: Pose estimator backend
            pose_model: Optional pose model path
            profiles_dir: Directory for fencer profiles
            llm_model_name: CoachEngine LLM model name
            ui_width: UI window width
            ui_height: UI window height
            create_ui: Whether to instantiate the OpenCV UI immediately
        """
        if device == "auto":
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(f"Initializing AI Fencing Coach on device: {device}")

        self.ui = None
        self.ui_width = ui_width
        self.ui_height = ui_height

        self.pipeline = SystemPipeline(
            use_bifencenet=use_bifencenet,
            device=device,
            model_checkpoint=model_checkpoint,
            profiles_dir=profiles_dir,
            pose_backend=pose_backend,
            pose_model_path=pose_model,
            llm_model_name=llm_model_name
        )

        if create_ui:
            self._ensure_ui()

        logger.info("Application initialized successfully")

    def _ensure_ui(self):
        """Lazy-create the OpenCV UI only when interactive mode needs it."""
        if self.ui is None:
            from src.app_interface.main_ui import FencingCoachUI

            self.ui = FencingCoachUI(
                width=self.ui_width,
                height=self.ui_height
            )
        return self.ui

    @staticmethod
    def _error_result(video_path: Path, fencer_id: str, error: str) -> Dict[str, Any]:
        """Build a stable error payload for CLI and tests."""
        return {
            "ok": False,
            "error": error,
            "video_path": str(video_path),
            "fencer_id": fencer_id,
            "frames_processed": 0,
            "classifications": [],
            "statistics": {},
            "feedback": "",
        }

    def process_video(
        self,
        video_path: str,
        fencer_id: str,
        opponent_id: Optional[str] = None,
        fencer_name: str = "",
        opponent_name: str = "",
        raise_on_error: bool = False
    ) -> Dict[str, Any]:
        """
        Process a fencing bout video.

        Args:
            video_path: Path to video file
            fencer_id: Fencer identifier string
            opponent_id: Opponent identifier (optional)
            fencer_name: Fencer full name (optional)
            opponent_name: Opponent full name (optional)
            raise_on_error: Re-raise processing failures for debugging

        Returns:
            Result dictionary with ok/error/feedback fields
        """
        video_file = Path(video_path).expanduser()
        if not video_file.exists():
            error = f"Video file not found: {video_file}"
            logger.error(error)
            return self._error_result(video_file, fencer_id, error)

        logger.info(f"Processing video: {video_file}")

        try:
            self.pipeline.set_fencer(fencer_id, fencer_name)
            if opponent_id:
                self.pipeline.set_opponent(opponent_id, opponent_name)

            results = self.pipeline.process_video(str(video_file), fencer_id)
            logger.info(f"Bout Statistics: {results['statistics']}")

            feedback = self.pipeline.get_conclusive_feedback(bout_result="completed")
            logger.info(f"Feedback: {feedback}")

            results["ok"] = True
            results["feedback"] = feedback
            return results
        except Exception as e:
            if raise_on_error:
                raise

            logger.exception("Error processing video")
            return self._error_result(video_file, fencer_id, str(e))

    def get_model_status(self) -> Dict[str, Any]:
        """Return action-recognition model checkpoint status."""
        return self.pipeline.get_model_status()

    def run_interactive_mode(self):
        """Run interactive UI for real-time coaching."""
        logger.info("Starting interactive mode...")
        ui = self._ensure_ui()

        print("\n" + "=" * 60)
        print("AI Fencing Coach & Referee System - Interactive Mode")
        print("=" * 60)

        try:
            while True:
                print("\nOptions:")
                print("1. Process video file")
                print("2. Configure fencer")
                print("3. View fencer profile")
                print("4. Exit")

                choice = input("Select option (1-4): ").strip()

                if choice == "1":
                    video_path = input("Enter video path: ").strip()
                    fencer_id = input("Enter fencer ID: ").strip()
                    opponent_id = input("Enter opponent ID (optional): ").strip() or None

                    results = self.process_video(video_path, fencer_id, opponent_id)
                    if not results.get("ok", False):
                        print(results["error"])

                elif choice == "2":
                    fencer_id = input("Enter fencer ID: ").strip()
                    fencer_name = input("Enter fencer name: ").strip()
                    self.pipeline.set_fencer(fencer_id, fencer_name)
                    print(f"Fencer configured: {fencer_id}")

                elif choice == "3":
                    fencer_id = input("Enter fencer ID: ").strip()
                    profile = self.pipeline.profile_manager.load_profile(fencer_id)
                    if profile:
                        print(f"\nProfile for {fencer_id}:")
                        print(f"  Name: {profile.get('name', 'N/A')}")
                        total_bouts = profile["overall_stats"].get("total_bouts", 0)
                        wins = profile["overall_stats"].get("wins", 0)
                        print(f"  Total Bouts: {total_bouts}")
                        print(f"  Wins: {wins}")
                    else:
                        print(f"No profile found for {fencer_id}")

                elif choice == "4":
                    print("Exiting...")
                    return
                else:
                    print("Invalid option")
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
        finally:
            ui.close()


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description="AI Fencing Coach & Referee System"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to YAML config file. Defaults to config.yaml when present."
    )
    parser.add_argument(
        "--video",
        type=str,
        help="Path to video file"
    )
    parser.add_argument(
        "--fencer-id",
        type=str,
        help="Fencer identifier. Defaults to athlete.default_id from config."
    )
    parser.add_argument(
        "--fencer-name",
        type=str,
        default="",
        help="Fencer full name"
    )
    parser.add_argument(
        "--opponent-id",
        type=str,
        help="Opponent identifier"
    )
    parser.add_argument(
        "--opponent-name",
        type=str,
        default="",
        help="Opponent full name"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["fencenet", "bifencenet"],
        help="Model architecture. Overrides model.type from config."
    )
    parser.add_argument(
        "--use-bifencenet",
        action="store_true",
        help="Use BiFenceNet instead of FenceNet"
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cuda", "cpu"],
        help="Device to use for inference"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Path to pretrained model checkpoint"
    )
    parser.add_argument(
        "--profiles-dir",
        type=str,
        help="Directory for fencer profiles"
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        help="CoachEngine LLM model name"
    )
    parser.add_argument(
        "--pose-backend",
        type=str,
        choices=["auto", "ultralytics", "mock"],
        help="Pose estimator backend"
    )
    parser.add_argument(
        "--pose-model",
        type=str,
        help="Path to YOLO pose model weights"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode"
    )

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Main entry point."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    try:
        config = load_app_config(_resolve_config_path(args.config))
    except (FileNotFoundError, ValueError, yaml.YAMLError) as e:
        logger.error(f"Configuration error: {e}")
        print(f"Configuration error: {e}")
        return 2

    model_type = args.model_type or _config_value(
        config, "model", "type", default="fencenet"
    )
    use_bifencenet = args.use_bifencenet or model_type == "bifencenet"
    device = args.device or _config_value(config, "llm", "device", default="auto")
    profiles_dir = args.profiles_dir or _config_value(
        config, "data", "profiles_dir", default="data/fencer_profiles/"
    )
    llm_model_name = args.llm_model or _config_value(
        config, "llm", "model_name", default="llava-next"
    )
    pose_backend = args.pose_backend or _config_value(
        config, "pose", "backend", default="auto"
    )
    pose_model = args.pose_model or _config_value(
        config, "pose", "model", default=None
    )
    fencer_id = args.fencer_id or _config_value(
        config, "athlete", "default_id", default="fencer_001"
    )
    ui_width = int(_config_value(config, "ui", "window_width", default=1600))
    ui_height = int(_config_value(config, "ui", "window_height", default=900))

    video_path = Path(args.video).expanduser() if args.video else None
    if video_path is not None and not video_path.exists():
        error = f"Video file not found: {video_path}"
        logger.error(error)
        print(error)
        return 1

    model_checkpoint = Path(args.model).expanduser() if args.model else None
    if model_checkpoint is not None and not model_checkpoint.exists():
        logger.warning(
            "Model checkpoint not found; continuing with randomly initialized "
            f"model: {model_checkpoint}"
        )

    try:
        app = FencingCoachApplication(
            use_bifencenet=use_bifencenet,
            device=device,
            model_checkpoint=model_checkpoint,
            pose_backend=pose_backend,
            pose_model=pose_model,
            profiles_dir=profiles_dir,
            llm_model_name=llm_model_name,
            ui_width=ui_width,
            ui_height=ui_height,
            create_ui=args.interactive or video_path is None
        )
    except Exception as e:
        logger.exception("Application initialization failed")
        print(f"Application initialization failed: {e}")
        return 1

    if args.interactive or video_path is None:
        app.run_interactive_mode()
        return 0

    results = app.process_video(
        video_path=str(video_path),
        fencer_id=fencer_id,
        opponent_id=args.opponent_id,
        fencer_name=args.fencer_name,
        opponent_name=args.opponent_name
    )
    if not results.get("ok", False):
        print(results["error"])
        return 1

    print(f"Processed video: {results['video_path']}")
    print(f"Frames processed: {results.get('frames_processed', 0)}")
    model_status_getter = getattr(app, "get_model_status", None)
    if model_status_getter:
        print(_format_model_status(model_status_getter()))
    if results.get("feedback"):
        print(f"Feedback: {results['feedback']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
