"""Small no-dependency web demo for the fencing coach MVP."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from html import escape
import json
import logging
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import re
from typing import Any, Dict, Optional
from urllib.parse import parse_qs, urlparse

from app import (
    FencingCoachApplication,
    _as_int,
    _build_height_calibration,
    _optional_positive_int,
    write_json_report,
)
from src.app_interface.video_annotator import write_annotated_video

logger = logging.getLogger(__name__)

DEFAULT_WEB_OUTPUT_DIR = Path("web_outputs")
DEFAULT_VIDEO_PATH = Path("data/videos/fencing_match.mp4")
DEFAULT_ANNOTATED_MAX_WIDTH = 1280


@dataclass
class WebProcessRequest:
    """Video-processing options submitted from the browser demo."""

    video_path: str = str(DEFAULT_VIDEO_PATH)
    fencer_id: str = "web_demo_fencer"
    pose_backend: str = "ultralytics"
    pose_model: str = "yolov8n-pose.pt"
    device: str = "cpu"
    left_height_cm: Optional[float] = 170.0
    right_height_cm: Optional[float] = 185.0
    annotated_max_width: Optional[int] = DEFAULT_ANNOTATED_MAX_WIDTH


@dataclass
class WebProcessResult:
    """Processed-video outputs for rendering in the web demo."""

    ok: bool
    error: str = ""
    input_video_path: Optional[Path] = None
    annotated_video_path: Optional[Path] = None
    report_path: Optional[Path] = None
    report: Optional[Dict[str, Any]] = None
    results: Optional[Dict[str, Any]] = None


def run_server(
    host: str = "127.0.0.1",
    port: int = 7860,
    repo_root: Optional[Path] = None,
    output_dir: Optional[Path] = None,
):
    """Run the local browser demo server."""
    repo_root = Path(repo_root or Path.cwd()).resolve()
    if output_dir is None:
        output_dir = repo_root / DEFAULT_WEB_OUTPUT_DIR
    else:
        output_dir = Path(output_dir)
        if not output_dir.is_absolute():
            output_dir = repo_root / output_dir
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    class ConfiguredHandler(WebDemoHandler):
        pass

    ConfiguredHandler.repo_root = repo_root
    ConfiguredHandler.output_dir = output_dir

    server = ThreadingHTTPServer((host, port), ConfiguredHandler)
    url = f"http://{host}:{port}"
    logger.info("Web demo available at %s", url)
    print(f"AI Fencing Coach web demo: {url}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping web demo...")
    finally:
        server.server_close()


class WebDemoHandler(BaseHTTPRequestHandler):
    """HTTP handler for the local browser demo."""

    repo_root = Path.cwd()
    output_dir = Path.cwd() / DEFAULT_WEB_OUTPUT_DIR

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path.startswith("/outputs/"):
            self._serve_output(parsed.path.removeprefix("/outputs/"))
            return

        self._send_html(render_home_page())

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path != "/process":
            self.send_error(HTTPStatus.NOT_FOUND, "Unknown endpoint")
            return

        length = int(self.headers.get("Content-Length", "0") or 0)
        body = self.rfile.read(length).decode("utf-8", errors="replace")
        form = parse_qs(body, keep_blank_values=True)
        try:
            request = request_from_form(form)
            result = process_web_video(
                request,
                repo_root=self.repo_root,
                output_dir=self.output_dir,
            )
        except ValueError as exc:
            request = WebProcessRequest()
            result = WebProcessResult(ok=False, error=str(exc))
        self._send_html(render_home_page(result=result, defaults=request))

    def log_message(self, format_string: str, *args):
        logger.info("web-demo: " + format_string, *args)

    def _send_html(self, html: str):
        encoded = html.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _serve_output(self, output_name: str):
        safe_name = Path(output_name).name
        output_path = (self.output_dir / safe_name).resolve()
        if self.output_dir not in output_path.parents or not output_path.exists():
            self.send_error(HTTPStatus.NOT_FOUND, "Output file not found")
            return

        if output_path.suffix.lower() == ".mp4":
            content_type = "video/mp4"
        elif output_path.suffix.lower() == ".json":
            content_type = "application/json"
        else:
            content_type = "application/octet-stream"

        self._serve_file(output_path, content_type)

    def _serve_file(self, output_path: Path, content_type: str):
        file_size = output_path.stat().st_size
        range_header = self.headers.get("Range")
        start = 0
        end = file_size - 1
        status = HTTPStatus.OK

        if range_header and range_header.startswith("bytes="):
            start_text, _, end_text = range_header.removeprefix("bytes=").partition("-")
            try:
                start = int(start_text) if start_text else 0
                end = int(end_text) if end_text else file_size - 1
                end = min(end, file_size - 1)
                if start < 0 or start > end:
                    raise ValueError
                status = HTTPStatus.PARTIAL_CONTENT
            except ValueError:
                self.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                return

        content_length = end - start + 1
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Accept-Ranges", "bytes")
        if status == HTTPStatus.PARTIAL_CONTENT:
            self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
        self.send_header("Content-Length", str(content_length))
        self.end_headers()

        with output_path.open("rb") as output_file:
            output_file.seek(start)
            remaining = content_length
            while remaining > 0:
                chunk = output_file.read(min(64 * 1024, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)


def request_from_form(form: Dict[str, Any]) -> WebProcessRequest:
    """Build a typed request from URL-encoded form values."""
    return WebProcessRequest(
        video_path=_form_value(form, "video_path", str(DEFAULT_VIDEO_PATH)),
        fencer_id=_form_value(form, "fencer_id", "web_demo_fencer"),
        pose_backend=_form_value(form, "pose_backend", "ultralytics"),
        pose_model=_form_value(form, "pose_model", "yolov8n-pose.pt"),
        device=_form_value(form, "device", "cpu"),
        left_height_cm=_optional_float(_form_value(form, "left_height_cm", "")),
        right_height_cm=_optional_float(_form_value(form, "right_height_cm", "")),
        annotated_max_width=_optional_int(
            _form_value(form, "annotated_max_width", str(DEFAULT_ANNOTATED_MAX_WIDTH))
        ),
    )


def process_web_video(
    request: WebProcessRequest,
    repo_root: Path,
    output_dir: Path,
) -> WebProcessResult:
    """Run the existing app pipeline and return web-renderable outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    profiles_dir = output_dir / "profiles"
    profiles_dir.mkdir(parents=True, exist_ok=True)

    try:
        video_path = _resolve_video_path(request.video_path, repo_root=repo_root)
        if not video_path.exists():
            return WebProcessResult(ok=False, error=f"Video not found: {video_path}")

        max_width = _optional_positive_int(
            request.annotated_max_width,
            "annotated max width",
        )
        heights = _build_height_calibration(
            left_height_cm=request.left_height_cm,
            right_height_cm=request.right_height_cm,
        )
        run_id = _run_id(video_path=video_path, fencer_id=request.fencer_id)
        annotated_video_path = output_dir / f"{run_id}_processed.mp4"
        report_path = output_dir / f"{run_id}_report.json"

        app = FencingCoachApplication(
            use_bifencenet=False,
            device=request.device,
            pose_backend=request.pose_backend,
            pose_model=request.pose_model or None,
            profiles_dir=str(profiles_dir),
            create_ui=False,
        )
        results = app.process_video(
            video_path=str(video_path),
            fencer_id=request.fencer_id,
        )
        if not results.get("ok", False):
            return WebProcessResult(
                ok=False,
                error=str(results.get("error", "Video processing failed")),
                input_video_path=video_path,
                results=results,
            )

        runtime_getter = getattr(app, "get_runtime_metadata", None)
        runtime_metadata = runtime_getter() if runtime_getter else {}
        written_report = write_json_report(
            results,
            output_path=report_path,
            reports_dir=str(output_dir),
            runtime_metadata=runtime_metadata,
        )
        written_video = write_annotated_video(
            str(video_path),
            output_path=annotated_video_path,
            tracking_frames=(results.get("two_fencer_tracking") or {}).get("frames", []),
            classifications=results.get("classifications", []),
            window_size=_as_int(results.get("window_size"), default=28),
            window_stride=_as_int(results.get("window_stride"), default=14),
            fencer_heights_cm=heights,
            max_width=max_width,
        )
        report = json.loads(written_report.read_text(encoding="utf-8"))
        return WebProcessResult(
            ok=True,
            input_video_path=video_path,
            annotated_video_path=written_video,
            report_path=written_report,
            report=report,
            results=results,
        )
    except Exception as exc:
        logger.exception("Web demo processing failed")
        return WebProcessResult(ok=False, error=str(exc))


def render_home_page(
    result: Optional[WebProcessResult] = None,
    defaults: Optional[WebProcessRequest] = None,
) -> str:
    """Render the single-page web demo."""
    defaults = defaults or WebProcessRequest()
    result_html = _render_result(result) if result else ""
    return f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>AI Fencing Coach Demo</title>
  <style>{_css()}</style>
</head>
<body>
  <main>
    <section class=\"hero\">
      <div>
        <p class=\"eyebrow\">AI Fencing Coach MVP</p>
        <h1>Visual bout review without terminal commands</h1>
        <p>Choose a server-side video, set fencer heights, run real or mock pose, then review the annotated HUD video and summary report in the browser.</p>
      </div>
    </section>

    <section class=\"card\">
      <h2>Process Video</h2>
      <form action=\"/process\" method=\"post\">
        <label>Video path on this machine
          <input name=\"video_path\" value=\"{escape(defaults.video_path)}\" required>
        </label>
        <div class=\"grid\">
          <label>Fencer ID
            <input name=\"fencer_id\" value=\"{escape(defaults.fencer_id)}\" required>
          </label>
          <label>Device
            <select name=\"device\">{_option(defaults.device, 'cpu')}{_option(defaults.device, 'cuda')}{_option(defaults.device, 'auto')}</select>
          </label>
        </div>
        <div class=\"grid\">
          <label>Pose backend
            <select name=\"pose_backend\">{_option(defaults.pose_backend, 'ultralytics')}{_option(defaults.pose_backend, 'mock')}{_option(defaults.pose_backend, 'auto')}</select>
          </label>
          <label>Pose model
            <input name=\"pose_model\" value=\"{escape(defaults.pose_model)}\">
          </label>
        </div>
        <div class=\"grid\">
          <label>Left height cm
            <input name=\"left_height_cm\" type=\"number\" step=\"0.1\" value=\"{_value(defaults.left_height_cm)}\">
          </label>
          <label>Right height cm
            <input name=\"right_height_cm\" type=\"number\" step=\"0.1\" value=\"{_value(defaults.right_height_cm)}\">
          </label>
          <label>Annotated max width (px)
            <input name=\"annotated_max_width\" type=\"number\" step=\"1\" value=\"{_value(defaults.annotated_max_width)}\">
          </label>
        </div>
        <button type=\"submit\">Process Video</button>
        <p class=\"note\">Tip: use <code>ultralytics</code> for real CV boxes. <code>mock</code> is deterministic and will not follow real fencers.</p>
        <p class=\"note\"><strong>Annotated max width</strong> downscales and H.264-transcodes the exported MP4 for smoother browser playback. It does not change pose detection, tracking, or inference.</p>
      </form>
    </section>

    {result_html}
  </main>
</body>
</html>"""


def _render_result(result: Optional[WebProcessResult]) -> str:
    if result is None:
        return ""
    if not result.ok:
        return f"""<section class=\"card error\"><h2>Processing Failed</h2><p>{escape(result.error)}</p></section>"""

    report = result.report or {}
    summary = ((report.get("two_fencer_tracking") or {}).get("summary") or {})
    statistics = report.get("statistics") or {}
    runtime = report.get("runtime") or {}
    video_name = result.annotated_video_path.name if result.annotated_video_path else ""
    report_name = result.report_path.name if result.report_path else ""
    action_frequencies = statistics.get("action_frequencies") or {}

    return f"""
    <section class=\"card\">
      <h2>Annotated Review</h2>
      <video controls preload=\"metadata\" src=\"/outputs/{escape(video_name)}\"></video>
      <p><a href=\"/outputs/{escape(report_name)}\" target=\"_blank\">Open JSON report</a></p>
    </section>
    <section class=\"cards\">
      {_metric_card('Frames', report.get('frames_processed', 0))}
      {_metric_card('Two-fencer coverage', _percent(summary.get('two_fencer_coverage')))}
      {_metric_card('Too-close ratio', _percent(summary.get('too_close_ratio')))}
      {_metric_card('Avg confidence', f"{float(statistics.get('average_confidence') or 0.0):.2f}")}
    </section>
    <section class=\"card\">
      <h2>Summary</h2>
      <p><strong>Pose backend:</strong> {escape(str(runtime.get('pose_backend', 'unknown')))}</p>
      <p><strong>Model weights:</strong> {escape(str(runtime.get('model_weights', 'unknown')))}</p>
      <p><strong>Top actions:</strong> {escape(_format_actions(action_frequencies))}</p>
      <p><strong>Feedback:</strong> {escape(str(report.get('feedback', '')))}</p>
      <p class=\"note\">Action labels remain prototype-level unless a trained FenceNet/BiFenceNet checkpoint is loaded.</p>
    </section>
    """


def _resolve_video_path(video_path: str, repo_root: Path) -> Path:
    candidate = Path(video_path).expanduser()
    if not candidate.is_absolute():
        candidate = repo_root / candidate
    return candidate.resolve()


def _run_id(video_path: Path, fencer_id: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = _safe_name(video_path.stem or "video")
    fencer = _safe_name(fencer_id or "fencer")
    return f"{timestamp}_{stem}_{fencer}"


def _safe_name(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return sanitized or "item"


def _form_value(form: Dict[str, Any], name: str, default: str) -> str:
    value = form.get(name, [default])
    if isinstance(value, list):
        return str(value[0]) if value else default
    return str(value)


def _optional_float(value: str) -> Optional[float]:
    if value is None or str(value).strip() == "":
        return None
    return float(value)


def _optional_int(value: str) -> Optional[int]:
    if value is None or str(value).strip() == "":
        return None
    return int(value)


def _option(current: str, value: str) -> str:
    selected = " selected" if current == value else ""
    return f"<option value=\"{escape(value)}\"{selected}>{escape(value)}</option>"


def _value(value: Any) -> str:
    return "" if value is None else escape(str(value))


def _metric_card(label: str, value: Any) -> str:
    return f"<div class=\"metric\"><span>{escape(label)}</span><strong>{escape(str(value))}</strong></div>"


def _percent(value: Any) -> str:
    try:
        return f"{float(value) * 100.0:.1f}%"
    except (TypeError, ValueError):
        return "0.0%"


def _format_actions(action_frequencies: Dict[str, Any]) -> str:
    if not action_frequencies:
        return "none"
    pairs = sorted(
        action_frequencies.items(),
        key=lambda item: float(item[1] or 0.0),
        reverse=True,
    )
    return ", ".join(
        f"{action}: {float(freq) * 100.0:.0f}%" for action, freq in pairs[:4]
    )


def _css() -> str:
    return """
    :root { color-scheme: dark; font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }
    body { margin: 0; background: #10131a; color: #f4f7fb; }
    main { max-width: 1180px; margin: 0 auto; padding: 28px; }
    .hero { display: flex; justify-content: space-between; gap: 24px; align-items: center; margin-bottom: 22px; }
    .hero h1 { font-size: clamp(2rem, 4vw, 4rem); line-height: 1; margin: 8px 0; }
    .hero p { color: #b8c2d1; max-width: 820px; }
    .eyebrow { color: #ffb45c; text-transform: uppercase; letter-spacing: .14em; font-weight: 700; }
    .card, .metric { background: #181d27; border: 1px solid #2b3445; border-radius: 18px; padding: 22px; box-shadow: 0 18px 50px rgb(0 0 0 / 25%); margin-bottom: 20px; }
    .cards { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 14px; }
    .metric { margin: 0; }
    .metric span { display: block; color: #9da9bb; font-size: .9rem; }
    .metric strong { display: block; font-size: 1.7rem; margin-top: 8px; }
    label { display: grid; gap: 8px; color: #c8d3e2; font-weight: 650; margin: 14px 0; }
    input, select { width: 100%; box-sizing: border-box; padding: 12px 14px; border-radius: 12px; border: 1px solid #3a465b; background: #10131a; color: #f4f7fb; font: inherit; }
    .grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 16px; }
    button { border: 0; border-radius: 999px; padding: 13px 22px; background: #ff8c2a; color: #171717; font-weight: 800; cursor: pointer; }
    video { display: block; width: 100%; max-height: 70vh; background: #05070a; border-radius: 12px; }
    a { color: #ffb45c; }
    code { background: #0b0e14; padding: 2px 6px; border-radius: 6px; }
    .note { color: #9da9bb; }
    .error { border-color: #ff5c5c; }
    @media (max-width: 800px) { .grid, .cards { grid-template-columns: 1fr; } main { padding: 16px; } }
    """
