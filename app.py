"""
AI Fencing Coach & Referee System - Main Application.

This is the entry point for the fencing coaching system.
It demonstrates the complete pipeline from video input to coaching feedback.
"""

import argparse
import json
import logging
from pathlib import Path
import re
from typing import Any, Dict, Optional, Sequence

import yaml

from src.app_interface.system_pipeline import SystemPipeline
from src.app_interface.video_annotator import write_annotated_video
from src.tracking import PatternAnalyzer

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


def _format_tracking_summary(summary: Dict[str, Any]) -> str:
    """Format two-fencer tracking coverage for CLI output."""
    frames = _as_int(summary.get("frames_analyzed"))
    frames_with_two = _as_int(summary.get("frames_with_two_fencers"))
    coverage = _as_float(summary.get("two_fencer_coverage")) * 100.0
    average_distance = summary.get("average_engagement_distance_px")
    frames_too_close = _as_int(summary.get("frames_too_close"))
    too_close_ratio = _as_float(summary.get("too_close_ratio")) * 100.0

    message = (
        "Two-fencer tracking: "
        f"{frames_with_two}/{frames} frames with two fencers "
        f"({coverage:.1f}% coverage)"
    )
    if average_distance is not None:
        message += f", avg front-ankle distance {float(average_distance):.1f}px"
    if frames_too_close:
        message += f"; too close in {frames_too_close} frames ({too_close_ratio:.1f}%)"
    return message


def _config_bool(value: Any, default: bool = False) -> bool:
    """Parse optional YAML/CLI-style booleans predictably."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return bool(value)


def _safe_filename_component(value: Any, fallback: str) -> str:
    """Return a filesystem-safe component for generated report filenames."""
    component = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or fallback))
    component = component.strip("._")
    return component or fallback


def _as_int(value: Any, default: int = 0) -> int:
    """Convert JSON-ish numeric values to int without leaking numpy scalars."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, default: float = 0.0) -> float:
    """Convert JSON-ish numeric values to float without leaking numpy scalars."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _optional_positive_float(value: Any, label: str) -> Optional[float]:
    """Parse optional positive numeric CLI/config values."""
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a number") from exc
    if parsed <= 0:
        raise ValueError(f"{label} must be greater than 0")
    return parsed


def _build_height_calibration(
    left_height_cm: Any = None,
    right_height_cm: Any = None
) -> Dict[str, float]:
    """Build optional per-side height calibration for annotated videos."""
    calibration: Dict[str, float] = {}
    left_height = _optional_positive_float(left_height_cm, "left height")
    right_height = _optional_positive_float(right_height_cm, "right height")
    if left_height is not None:
        calibration["fencer_L"] = left_height
    if right_height is not None:
        calibration["fencer_R"] = right_height
    return calibration


def _format_height_calibration(calibration: Dict[str, float]) -> str:
    """Format optional fencer height calibration for CLI output."""
    if not calibration:
        return "Fencer height calibration: auto from detected bbox height"
    parts = [
        f"{track_id}={height_cm:.0f}cm"
        for track_id, height_cm in sorted(calibration.items())
    ]
    return "Fencer height calibration: " + ", ".join(parts)


def _json_default(value: Any) -> Any:
    """Handle values json.dump cannot serialize by default."""
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def build_video_report(
    results: Dict[str, Any],
    runtime_metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Build a stable JSON report payload for a processed video."""
    window_size = _as_int(results.get("window_size"), default=28)
    window_stride = _as_int(results.get("window_stride"), default=14)
    classification_windows = []

    for window_index, classification in enumerate(results.get("classifications", [])):
        class_idx, confidence = classification
        class_idx_int = _as_int(class_idx, default=-1)
        start_frame = window_index * window_stride
        classification_windows.append({
            "window_index": window_index,
            "window_start_frame": start_frame,
            "window_end_frame_exclusive": start_frame + window_size,
            "class_idx": class_idx_int,
            "action": PatternAnalyzer.ACTION_CLASSES.get(class_idx_int, "unknown"),
            "confidence": _as_float(confidence),
        })

    statistics = results.get("statistics") or {}
    tracking_payload = results.get("two_fencer_tracking") or {}
    tracking_report = {
        "schema_version": _as_int(
            tracking_payload.get("schema_version"),
            default=1
        ),
        "strategy": str(tracking_payload.get("strategy", "")),
        "identity_persistence": str(
            tracking_payload.get("identity_persistence", "")
        ),
        "too_close_rule": str(tracking_payload.get("too_close_rule", "")),
        "summary": tracking_payload.get("summary", {}),
        "frames": tracking_payload.get("frames", []),
    }

    return {
        "schema_version": 1,
        "ok": bool(results.get("ok", False)),
        "video_path": str(results.get("video_path", "")),
        "fencer_id": str(results.get("fencer_id", "")),
        "opponent_id": results.get("opponent_id"),
        "frames_processed": _as_int(results.get("frames_processed")),
        "window_size": window_size,
        "window_stride": window_stride,
        "classification_window_count": len(classification_windows),
        "classification_windows": classification_windows,
        "two_fencer_tracking": tracking_report,
        "statistics": {
            "total_actions": _as_int(
                statistics.get("total_actions"),
                default=len(classification_windows)
            ),
            "action_frequencies": statistics.get("action_frequencies", {}),
            "offensive_ratio": _as_float(statistics.get("offensive_ratio")),
            "defensive_ratio": _as_float(statistics.get("defensive_ratio")),
            "js_sf_ratio": _as_float(statistics.get("js_sf_ratio")),
            "repetitive_patterns": statistics.get("repetitive_patterns", []),
            "average_confidence": _as_float(
                statistics.get("average_confidence")
            ),
        },
        "feedback": str(results.get("feedback", "")),
        "runtime": runtime_metadata or {},
    }


def _default_report_path(results: Dict[str, Any], reports_dir: str) -> Path:
    """Build the default report path for a processed video result."""
    video_stem = _safe_filename_component(
        Path(str(results.get("video_path") or "video")).stem,
        "video"
    )
    fencer_id = _safe_filename_component(results.get("fencer_id"), "fencer")
    return Path(reports_dir).expanduser() / f"{video_stem}_{fencer_id}_report.json"


def write_json_report(
    results: Dict[str, Any],
    output_path: Optional[Path] = None,
    reports_dir: str = "reports/",
    runtime_metadata: Optional[Dict[str, Any]] = None
) -> Path:
    """Write a processed-video JSON report and return the written path."""
    report_path = (
        Path(output_path).expanduser()
        if output_path is not None
        else _default_report_path(results, reports_dir)
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = build_video_report(results, runtime_metadata=runtime_metadata)

    with report_path.open("w", encoding="utf-8") as report_file:
        json.dump(report, report_file, indent=2, sort_keys=True, default=_json_default)
        report_file.write("\n")

    return report_path


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
            results["opponent_id"] = opponent_id
            return results
        except Exception as e:
            if raise_on_error:
                raise

            logger.exception("Error processing video")
            return self._error_result(video_file, fencer_id, str(e))

    def get_model_status(self) -> Dict[str, Any]:
        """Return action-recognition model checkpoint status."""
        return self.pipeline.get_model_status()

    def get_runtime_metadata(self) -> Dict[str, Any]:
        """Return JSON-friendly runtime metadata for reports."""
        return self.pipeline.get_runtime_metadata()

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
        "--report",
        type=str,
        help="Path to write a JSON report for a processed video"
    )
    parser.add_argument(
        "--reports-dir",
        type=str,
        help="Directory for auto-named JSON reports when enabled by config"
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Disable config-driven JSON report writing"
    )
    parser.add_argument(
        "--annotated-video",
        type=str,
        help="Path to write an annotated MP4 with fencer boxes and distance cues"
    )
    parser.add_argument(
        "--left-height-cm",
        type=float,
        help="Optional left fencer height in centimeters for annotated-video HUD calibration"
    )
    parser.add_argument(
        "--right-height-cm",
        type=float,
        help="Optional right fencer height in centimeters for annotated-video HUD calibration"
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
    report_path = Path(args.report).expanduser() if args.report else None
    annotated_video_config = _config_value(
        config, "output", "annotated_video", default=None
    )
    annotated_video_value = args.annotated_video or annotated_video_config
    annotated_video_path = (
        Path(annotated_video_value).expanduser()
        if annotated_video_value
        else None
    )
    left_height_value = (
        args.left_height_cm
        if args.left_height_cm is not None
        else _config_value(config, "tracking", "left_height_cm", default=None)
    )
    right_height_value = (
        args.right_height_cm
        if args.right_height_cm is not None
        else _config_value(config, "tracking", "right_height_cm", default=None)
    )
    try:
        height_calibration = _build_height_calibration(
            left_height_cm=left_height_value,
            right_height_cm=right_height_value
        )
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        print(f"Configuration error: {e}")
        return 2
    reports_dir = args.reports_dir or _config_value(
        config, "output", "reports_dir", default="reports/"
    )
    save_report_from_config = _config_bool(
        _config_value(config, "output", "save_reports", default=False)
    )
    should_write_report = (
        report_path is not None
        or (save_report_from_config and not args.no_report)
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
    tracking_summary = (results.get("two_fencer_tracking") or {}).get("summary")
    if tracking_summary:
        print(_format_tracking_summary(tracking_summary))
    if annotated_video_path is not None:
        print(_format_height_calibration(height_calibration))
    model_status_getter = getattr(app, "get_model_status", None)
    if model_status_getter:
        print(_format_model_status(model_status_getter()))
    if results.get("feedback"):
        print(f"Feedback: {results['feedback']}")
    if should_write_report:
        try:
            metadata_getter = getattr(app, "get_runtime_metadata", None)
            runtime_metadata = metadata_getter() if metadata_getter else {}
            written_report = write_json_report(
                results,
                output_path=report_path,
                reports_dir=reports_dir,
                runtime_metadata=runtime_metadata
            )
        except OSError as e:
            logger.error(f"Could not write JSON report: {e}")
            print(f"Could not write JSON report: {e}")
            return 1
        print(f"Report written: {written_report}")

    if annotated_video_path is not None:
        try:
            written_video = write_annotated_video(
                str(video_path),
                output_path=annotated_video_path,
                tracking_frames=(
                    results.get("two_fencer_tracking") or {}
                ).get("frames", []),
                classifications=results.get("classifications", []),
                window_size=_as_int(results.get("window_size"), default=28),
                window_stride=_as_int(results.get("window_stride"), default=14),
                fencer_heights_cm=height_calibration
            )
        except (OSError, ValueError) as e:
            logger.error(f"Could not write annotated video: {e}")
            print(f"Could not write annotated video: {e}")
            return 1
        print(f"Annotated video written: {written_video}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
