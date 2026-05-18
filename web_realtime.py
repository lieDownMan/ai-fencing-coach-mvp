import json
import os
import cv2
import numpy as np
import uvicorn
from html import escape
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from pydantic import BaseModel

from inference.heuristic_debug import HEURISTIC_KEYS, compute_heuristic_metric, format_metrics, format_value
from src.realtime.feedback_scheduler import DEFAULT_ERROR_WEIGHTS, normalize_error_keys
from src.realtime.realtime_app import LiveVideoPipeline

app = FastAPI()

# Config
CAMERA_SOURCE = "0"  # Can be 0, 1, or URL like "http://192.168.x.x:4747/video"
TARGET_SIDE = "left"
TRAINING_MODE = "Footwork"
DEBUG_ENABLED = True
DEBUG_HEURISTIC = "all"
DEBUG_WINDOW_SIZE = 28
FEEDBACK_FOCUS_ERRORS = []
FEEDBACK_MUTE_ERRORS = []
FEEDBACK_ONLY_SELECTED = False

pipeline = None
cap = None

_PLAYBOOK_PATH = Path(__file__).resolve().parent / "coach_playbook.json"


def _load_playbook() -> dict:
    if not _PLAYBOOK_PATH.exists():
        return {}
    with open(_PLAYBOOK_PATH, "r", encoding="utf-8") as handle:
        return json.load(handle)


_PLAYBOOK = _load_playbook()

class ConfigUpdate(BaseModel):
    camera_source: str
    target_side: str
    training_mode: str
    debug_enabled: bool = True
    debug_heuristic: str = "all"
    debug_window_size: int = 28
    focus_errors: list[str] = []
    mute_errors: list[str] = []
    only_selected: bool = False

HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>AI Fencing Coach - Live</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background-color: #1a1a1a;
            color: white;
            text-align: center;
            margin: 0;
            padding: 20px;
        }
        h1 { margin-bottom: 10px; }
        .info { color: #aaa; font-size: 0.9em; margin-bottom: 15px; }
        
        .video-container {
            position: relative;
            display: inline-block;
            border: 4px solid #333;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 4px 15px rgba(0,0,0,0.5);
            background: black;
            max-width: 100%;
        }
        img {
            max-width: 100%;
            height: auto;
            display: block;
        }

        /* HUD Overlays (HTML instead of drawn on OpenCV frame) */
        .hud {
            position: absolute;
            left: 20px;
            text-align: left;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.8);
            pointer-events: none; /* Let clicks pass through to video */
        }
        
        #hud-state {
            top: 20px;
            color: #00ff00;
            font-size: 1.2em;
            font-weight: bold;
        }
        
        #hud-action {
            top: 60px;
            color: #00a5ff;
            font-size: 2em;
            font-weight: bold;
        }

        #hud-warnings {
            top: 110px;
            font-weight: bold;
            max-width: min(900px, calc(100vw - 80px));
        }
        .warning-primary {
            color: #ff4d4d;
            font-size: 2.35em;
            line-height: 1.08;
        }
        .warning-secondary {
            color: #ffd166;
            font-size: 1.25em;
            line-height: 1.25;
            margin-top: 8px;
        }
        .warning-score {
            color: #ccc;
            font-size: 0.55em;
            margin-left: 10px;
        }

        .controls {
            margin-top: 20px;
            padding: 20px;
            background: #2a2a2a;
            border: 1px solid #444;
            border-radius: 8px;
            display: inline-block;
            text-align: left;
        }
        .form-group {
            margin-bottom: 15px;
        }
        label {
            display: inline-block;
            width: 120px;
            font-weight: bold;
        }
        select, input[type="text"] {
            padding: 5px;
            background: #444;
            color: white;
            border: 1px solid #555;
            border-radius: 4px;
            width: 200px;
        }
        button {
            padding: 10px 20px;
            background: #ff6600;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-weight: bold;
            width: 100%;
        }
        button:hover {
            background: #ff8533;
        }
        input[type="checkbox"] {
            transform: scale(1.2);
            margin-right: 8px;
        }
        input[type="number"] {
            padding: 5px;
            background: #444;
            color: white;
            border: 1px solid #555;
            border-radius: 4px;
            width: 80px;
        }
        .secondary-button {
            background: #555;
            margin-top: 10px;
        }
        .secondary-button:hover {
            background: #777;
        }
        .feedback-section {
            margin-bottom: 15px;
            max-width: 620px;
        }
        .feedback-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 10px;
            margin-bottom: 8px;
        }
        .feedback-header label {
            width: auto;
        }
        .mini-button {
            width: auto;
            padding: 6px 10px;
            background: #555;
            font-size: 0.82em;
        }
        .mini-button:hover {
            background: #777;
        }
        .error-check-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(240px, 1fr));
            gap: 8px;
        }
        .error-option {
            width: auto;
            min-height: 44px;
            display: flex;
            align-items: flex-start;
            gap: 8px;
            padding: 8px;
            border: 1px solid #444;
            border-radius: 6px;
            background: #242424;
            cursor: pointer;
            font-weight: 400;
        }
        .error-option:hover {
            border-color: #777;
            background: #303030;
        }
        .error-option input[type="checkbox"] {
            margin-top: 4px;
            flex: 0 0 auto;
        }
        .error-label {
            display: block;
            color: #fff;
            line-height: 1.2;
        }
        .error-key {
            display: block;
            color: #aaa;
            font-family: Consolas, monospace;
            font-size: 0.82em;
            line-height: 1.2;
            margin-top: 2px;
        }
        @media (max-width: 760px) {
            .error-check-grid {
                grid-template-columns: 1fr;
            }
        }
        .debug-panel {
            margin: 20px auto 0;
            padding: 16px;
            background: #202020;
            border: 1px solid #444;
            border-radius: 8px;
            max-width: 1180px;
            text-align: left;
        }
        .debug-panel h3 {
            margin-top: 0;
        }
        .debug-meta {
            color: #bbb;
            margin-bottom: 10px;
            font-size: 0.95em;
        }
        .metric-row {
            display: grid;
            grid-template-columns: 210px 92px 135px 1fr;
            gap: 10px;
            align-items: start;
            padding: 8px;
            border-bottom: 1px solid #333;
            font-family: Consolas, monospace;
            font-size: 0.92em;
        }
        .metric-row.triggered {
            background: #5a2525;
            color: #fff;
        }
        .metric-row.ok {
            background: #242424;
        }
        .metric-key {
            font-weight: 700;
        }
        .metric-status {
            font-weight: 700;
        }
    </style>
</head>
<body>
    <h1>AI Fencing Coach Live</h1>
    <div class="info">
        Ensure your full body is visible. Voice coaching is enabled on the server side.
    </div>
    
    <div class="video-container">
        <!-- This img tag requests the MJPEG stream -->
        <img id="video-stream" src="/video_feed" alt="Video stream loading or camera not found..." />
        
        <!-- Dynamic HUD elements -->
        <div id="hud-state" class="hud">State: UNKNOWN</div>
        <div id="hud-action" class="hud">Action: Idle</div>
        <div id="hud-warnings" class="hud"></div>
    </div>

    <br>

    <div class="controls">
        <h3>Settings</h3>
        <div class="form-group">
            <label for="camera_source">Camera Source:</label>
            <input type="text" id="camera_source" value="{CAMERA_SOURCE}" placeholder="0 or http://IP:4747/video">
        </div>
        <div class="form-group">
            <label for="target_side">Target Side:</label>
            <select id="target_side">
                <option value="left" {TARGET_LEFT_SEL}>Left Fencer</option>
                <option value="right" {TARGET_RIGHT_SEL}>Right Fencer</option>
            </select>
        </div>
        <div class="form-group">
            <label for="training_mode">Training Mode:</label>
            <select id="training_mode">
                <option value="Footwork" {MODE_FW_SEL}>Footwork</option>
                <option value="Target Practice" {MODE_TP_SEL}>Target Practice</option>
                <option value="Free Bouting" {MODE_FB_SEL}>Free Bouting</option>
            </select>
        </div>
        <div class="form-group">
            <label for="debug_enabled">Debug:</label>
            <input type="checkbox" id="debug_enabled" {DEBUG_CHECKED}> Show heuristic metrics
        </div>
        <div class="form-group">
            <label for="debug_heuristic">Heuristic:</label>
            <select id="debug_heuristic">
                {HEURISTIC_OPTIONS}
            </select>
        </div>
        <div class="form-group">
            <label for="debug_window_size">Window:</label>
            <input type="number" id="debug_window_size" min="5" max="90" value="{DEBUG_WINDOW_SIZE}"> frames
        </div>
        <div class="feedback-section">
            <div class="feedback-header">
                <label>Focus Errors</label>
                <button type="button" class="mini-button" onclick="clearFeedbackChecks('focus_errors')">Clear</button>
            </div>
            <div id="focus_errors" class="error-check-grid">
                {FOCUS_ERROR_OPTIONS}
            </div>
        </div>
        <div class="feedback-section">
            <div class="feedback-header">
                <label>Mute Errors</label>
                <button type="button" class="mini-button" onclick="clearFeedbackChecks('mute_errors')">Clear</button>
            </div>
            <div id="mute_errors" class="error-check-grid">
                {MUTE_ERROR_OPTIONS}
            </div>
        </div>
        <div class="form-group">
            <label for="only_selected">Only:</label>
            <input type="checkbox" id="only_selected" {ONLY_SELECTED_CHECKED}> Only show/speak focused errors
        </div>
        <button onclick="updateConfig()">Apply & Restart Camera</button>
        <button class="secondary-button" onclick="resetTargetLock()">Reset Target Lock</button>
    </div>

    <div id="debug-panel" class="debug-panel">
        <h3>Realtime Heuristic Debug</h3>
        <div id="debug-meta" class="debug-meta">Waiting for pipeline...</div>
        <div id="debug-metrics"></div>
    </div>

    <script>
        function escapeHtml(value) {
            return String(value ?? '')
                .replaceAll('&', '&amp;')
                .replaceAll('<', '&lt;')
                .replaceAll('>', '&gt;')
                .replaceAll('"', '&quot;')
                .replaceAll("'", '&#039;');
        }

        function checkedValues(name) {
            return Array.from(document.querySelectorAll(`input[name="${name}"]:checked`)).map(input => input.value);
        }

        function clearFeedbackChecks(name) {
            document.querySelectorAll(`input[name="${name}"]`).forEach(input => {
                input.checked = false;
            });
        }

        document.addEventListener('change', event => {
            const input = event.target;
            if (!input || input.type !== 'checkbox' || !input.checked) {
                return;
            }
            if (input.name !== 'focus_errors' && input.name !== 'mute_errors') {
                return;
            }
            const oppositeName = input.name === 'focus_errors' ? 'mute_errors' : 'focus_errors';
            document.querySelectorAll(`input[name="${oppositeName}"]`).forEach(other => {
                if (other.value === input.value) {
                    other.checked = false;
                }
            });
        });

        // Poll the server for pipeline status every 200ms
        setInterval(async () => {
            try {
                const response = await fetch('/status');
                const data = await response.json();
                
                document.getElementById('hud-state').innerText = `State: ${data.state}`;
                document.getElementById('hud-action').innerText = `Action: ${data.action}`;
                
                // Update warnings
                const warningsDiv = document.getElementById('hud-warnings');
                if (data.feedback_items && data.feedback_items.length > 0) {
                    warningsDiv.innerHTML = data.feedback_items.map((item, index) => {
                        const cls = index === 0 ? 'warning-primary' : 'warning-secondary';
                        const score = Number(item.score ?? 0).toFixed(1);
                        return `<div class="${cls}">${escapeHtml(item.message)} <span class="warning-score">score ${score}</span></div>`;
                    }).join('');
                } else if (data.warnings && data.warnings.length > 0) {
                    warningsDiv.innerHTML = data.warnings.map(w => `<div class="warning-primary">${escapeHtml(w)}</div>`).join('');
                } else {
                    warningsDiv.innerHTML = '';
                }
                
                // Color formatting for State
                const stateEl = document.getElementById('hud-state');
                if (data.state === 'ACTIVE') {
                    stateEl.style.color = '#00ff00';
                } else if (data.state === 'IDLE') {
                    stateEl.style.color = '#ffaa00';
                } else {
                    stateEl.style.color = '#aaaaaa';
                }

                const panel = document.getElementById('debug-panel');
                const meta = document.getElementById('debug-meta');
                const metrics = document.getElementById('debug-metrics');
                if (!data.debug_enabled) {
                    panel.style.display = 'none';
                } else {
                    panel.style.display = 'block';
                    const lock = data.target_lock || {};
                    meta.innerText = `frame=${data.frame} | lock=${lock.mode || 'unknown'} | track_id=${lock.track_id ?? 'none'} | buffer=${data.debug_buffer}/${data.debug_window_size}`;
                    const rows = data.heuristic_metrics || [];
                    if (rows.length === 0) {
                        metrics.innerHTML = '<div class="debug-meta">No target skeleton yet.</div>';
                    } else {
                        metrics.innerHTML = rows.map(row => {
                            const cls = row.triggered ? 'triggered' : 'ok';
                            const status = row.triggered ? 'TRIGGER' : 'OK';
                            return `
                                <div class="metric-row ${cls}">
                                    <div class="metric-key">${escapeHtml(row.heuristic)}</div>
                                    <div class="metric-status">${status}</div>
                                    <div>value=${escapeHtml(row.primary_value)}</div>
                                    <div>${escapeHtml(row.metrics)}</div>
                                </div>
                            `;
                        }).join('');
                    }
                }
                
            } catch (err) {
                console.error("Error fetching status:", err);
            }
        }, 200);

        async function updateConfig() {
            const cam = document.getElementById('camera_source').value;
            const target = document.getElementById('target_side').value;
            const mode = document.getElementById('training_mode').value;
            const debugEnabled = document.getElementById('debug_enabled').checked;
            const debugHeuristic = document.getElementById('debug_heuristic').value;
            const debugWindowSize = parseInt(document.getElementById('debug_window_size').value || '28', 10);
            const focusErrors = checkedValues('focus_errors');
            const muteErrors = checkedValues('mute_errors');
            const onlySelected = document.getElementById('only_selected').checked;

            try {
                const res = await fetch('/config', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        camera_source: cam,
                        target_side: target,
                        training_mode: mode,
                        debug_enabled: debugEnabled,
                        debug_heuristic: debugHeuristic,
                        debug_window_size: debugWindowSize,
                        focus_errors: focusErrors,
                        mute_errors: muteErrors,
                        only_selected: onlySelected
                    })
                });
                
                if (res.ok) {
                    // Reload the image stream to force a reconnect with new settings
                    const img = document.getElementById('video-stream');
                    img.src = '/video_feed?' + new Date().getTime();
                    alert("Settings applied! Camera is restarting...");
                } else {
                    alert("Failed to update settings.");
                }
            } catch (e) {
                alert("Error updating settings.");
            }
        }

        async function resetTargetLock() {
            try {
                const res = await fetch('/reset_target', { method: 'POST' });
                if (res.ok) {
                    alert('Target lock reset.');
                } else {
                    alert('Failed to reset target lock.');
                }
            } catch (e) {
                alert('Error resetting target lock.');
            }
        }
    </script>
</body>
</html>
"""

def generate_frames():
    global cap, pipeline
    
    # Initialize only when someone connects
    if cap is None or not cap.isOpened():
        try:
            source = int(CAMERA_SOURCE)
        except ValueError:
            source = CAMERA_SOURCE
            
        print(f"Opening camera: {source}")
        cap = cv2.VideoCapture(source)
        
        if not cap.isOpened():
            print(f"Error: Cannot open video source {source}")
            # Yield a blank frame or error frame
            img = cv2.imread("web_outputs/placeholder.png") if os.path.exists("web_outputs/placeholder.png") else np.zeros((480, 640, 3), dtype=np.uint8)
            blank = cv2.imencode('.jpg', cv2.resize(img, (640, 480)), [cv2.IMWRITE_JPEG_QUALITY, 50])[1].tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + blank + b'\r\n')
            return

    if pipeline is None:
        print(f"Initializing AI Pipeline (Mode: {TRAINING_MODE}, Target: {TARGET_SIDE})...")
        pipeline = LiveVideoPipeline(
            target_side=TARGET_SIDE,
            training_mode=TRAINING_MODE,
            pose_backend="ultralytics",
            voice_enabled=True,
            focus_errors=FEEDBACK_FOCUS_ERRORS,
            mute_errors=FEEDBACK_MUTE_ERRORS,
            only_selected=FEEDBACK_ONLY_SELECTED,
        )

    drop_count = 0
    while True:
        # If pipeline gets reset externally, break generator to force reconnect
        if pipeline is None or cap is None:
            break

        ret, frame = cap.read()
        if not ret:
            drop_count += 1
            if drop_count > 30:
                print("Camera connection lost.")
                break
            continue
            
        drop_count = 0
        
        # AI Processing (Tell pipeline NOT to draw HUD text on the frame)
        out_frame = pipeline.process_frame(frame, draw_hud=False)
        
        # Encode as JPEG
        ret, buffer = cv2.imencode('.jpg', out_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ret:
            continue
            
        frame_bytes = buffer.tobytes()
        
        # Yield frame in MJPEG format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


def _heuristic_options_html() -> str:
    choices = ["all"] + HEURISTIC_KEYS
    return "\n".join(
        f'<option value="{choice}" {"selected" if choice == DEBUG_HEURISTIC else ""}>{choice}</option>'
        for choice in choices
    )


def _feedback_error_label(error_key: str) -> str:
    entry = _PLAYBOOK.get(error_key, {})
    return str(entry.get("error_name") or error_key)


def _feedback_error_checkboxes_html(group_name: str, selected_keys) -> str:
    selected = set(selected_keys or [])
    choices = sorted(DEFAULT_ERROR_WEIGHTS.keys())
    return "\n".join(
        "\n".join(
            [
                '<label class="error-option">',
                (
                    f'<input type="checkbox" name="{escape(group_name)}" '
                    f'value="{escape(choice)}" {"checked" if choice in selected else ""}>'
                ),
                "<span>",
                f'<span class="error-label">{escape(_feedback_error_label(choice))}</span>',
                f'<span class="error-key">{escape(choice)}</span>',
                "</span>",
                "</label>",
            ]
        )
        for choice in choices
    )


def _current_heuristic_metrics():
    if pipeline is None or not DEBUG_ENABLED:
        return []
    raw_skeletons = [
        skel for skel in list(getattr(pipeline, "raw_skeletons", []))[-max(1, DEBUG_WINDOW_SIZE):]
        if skel
    ]
    if not raw_skeletons:
        return []

    keys = HEURISTIC_KEYS if DEBUG_HEURISTIC == "all" else [DEBUG_HEURISTIC]
    rows = []
    for key in keys:
        metric = compute_heuristic_metric(
            key,
            raw_skeletons,
            target_side=TARGET_SIDE,
            training_mode=TRAINING_MODE,
        )
        rows.append({
            "heuristic": key,
            "triggered": metric.triggered,
            "primary_value": format_value(metric.primary_value),
            "threshold": metric.threshold,
            "metrics": format_metrics(metric),
        })
    rows.sort(key=lambda row: (not row["triggered"], row["heuristic"]))
    return rows


def _current_feedback_items():
    if pipeline is None:
        return []
    rows = []
    for item in getattr(pipeline, "current_feedback_items", []):
        if hasattr(item, "to_dict"):
            rows.append(item.to_dict())
        elif isinstance(item, dict):
            rows.append(dict(item))
    return rows


def _target_lock_status():
    if pipeline is None:
        return {"mode": "none", "track_id": None, "missing_frames": 0}
    tracker = pipeline.target_tracker
    return {
        "mode": "track_id" if tracker.locked_track_id is not None else "position",
        "track_id": tracker.locked_track_id,
        "missing_frames": tracker.missing_frames_count,
    }


@app.get("/")
def index():
    html = HTML_PAGE.replace("{CAMERA_SOURCE}", CAMERA_SOURCE)
    
    html = html.replace("{TARGET_LEFT_SEL}", "selected" if TARGET_SIDE == "left" else "")
    html = html.replace("{TARGET_RIGHT_SEL}", "selected" if TARGET_SIDE == "right" else "")
    
    html = html.replace("{MODE_FW_SEL}", "selected" if TRAINING_MODE == "Footwork" else "")
    html = html.replace("{MODE_TP_SEL}", "selected" if TRAINING_MODE == "Target Practice" else "")
    html = html.replace("{MODE_FB_SEL}", "selected" if TRAINING_MODE == "Free Bouting" else "")
    html = html.replace("{DEBUG_CHECKED}", "checked" if DEBUG_ENABLED else "")
    html = html.replace("{HEURISTIC_OPTIONS}", _heuristic_options_html())
    html = html.replace("{DEBUG_WINDOW_SIZE}", str(DEBUG_WINDOW_SIZE))
    html = html.replace("{FOCUS_ERROR_OPTIONS}", _feedback_error_checkboxes_html("focus_errors", FEEDBACK_FOCUS_ERRORS))
    html = html.replace("{MUTE_ERROR_OPTIONS}", _feedback_error_checkboxes_html("mute_errors", FEEDBACK_MUTE_ERRORS))
    html = html.replace("{ONLY_SELECTED_CHECKED}", "checked" if FEEDBACK_ONLY_SELECTED else "")

    return HTMLResponse(content=html)


@app.get("/video_feed")
def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/status")
def status():
    """Endpoint for frontend JS to poll current pipeline state dynamically."""
    if pipeline is None:
        return JSONResponse({
            "state": "INITIALIZING",
            "action": "Wait...",
            "warnings": [],
            "feedback_items": [],
            "debug_enabled": DEBUG_ENABLED,
            "debug_buffer": 0,
            "debug_window_size": DEBUG_WINDOW_SIZE,
            "heuristic_metrics": [],
            "feedback_preferences": {
                "focus_errors": FEEDBACK_FOCUS_ERRORS,
                "mute_errors": FEEDBACK_MUTE_ERRORS,
                "only_selected": FEEDBACK_ONLY_SELECTED,
            },
            "target_lock": {"mode": "none", "track_id": None, "missing_frames": 0},
            "frame": 0,
        })
        
    return JSONResponse({
        "state": getattr(pipeline.gatekeeper, "state", "UNKNOWN"),
        "action": pipeline.current_action,
        "warnings": pipeline.current_warnings if pipeline.warning_frames_left > 0 else [],
        "feedback_items": _current_feedback_items() if pipeline.warning_frames_left > 0 else [],
        "debug_enabled": DEBUG_ENABLED,
        "debug_buffer": len([skel for skel in list(pipeline.raw_skeletons)[-max(1, DEBUG_WINDOW_SIZE):] if skel]),
        "debug_window_size": DEBUG_WINDOW_SIZE,
        "heuristic_metrics": _current_heuristic_metrics(),
        "feedback_preferences": {
            "focus_errors": FEEDBACK_FOCUS_ERRORS,
            "mute_errors": FEEDBACK_MUTE_ERRORS,
            "only_selected": FEEDBACK_ONLY_SELECTED,
        },
        "target_lock": _target_lock_status(),
        "frame": pipeline.frame_idx,
    })


@app.post("/config")
def update_config(config: ConfigUpdate):
    """Endpoint to update global settings from the UI and reset the pipeline."""
    global CAMERA_SOURCE, TARGET_SIDE, TRAINING_MODE, DEBUG_ENABLED, DEBUG_HEURISTIC, DEBUG_WINDOW_SIZE
    global FEEDBACK_FOCUS_ERRORS, FEEDBACK_MUTE_ERRORS, FEEDBACK_ONLY_SELECTED, pipeline, cap
    
    CAMERA_SOURCE = config.camera_source
    TARGET_SIDE = config.target_side
    TRAINING_MODE = config.training_mode
    DEBUG_ENABLED = bool(config.debug_enabled)
    DEBUG_HEURISTIC = config.debug_heuristic if config.debug_heuristic in ["all"] + HEURISTIC_KEYS else "all"
    DEBUG_WINDOW_SIZE = max(5, min(90, int(config.debug_window_size)))
    valid_errors = set(DEFAULT_ERROR_WEIGHTS.keys())
    FEEDBACK_FOCUS_ERRORS = [
        key for key in normalize_error_keys(config.focus_errors)
        if key in valid_errors
    ]
    FEEDBACK_MUTE_ERRORS = [
        key for key in normalize_error_keys(config.mute_errors)
        if key in valid_errors
    ]
    FEEDBACK_ONLY_SELECTED = bool(config.only_selected)
    
    # Release current resources to trigger a re-init
    if cap is not None:
        cap.release()
    cap = None
    
    if pipeline is not None and pipeline.voice_coach is not None:
        pipeline.voice_coach.shutdown()
    pipeline = None
    
    return {"status": "success", "message": "Configuration updated. Restarting pipeline."}


@app.post("/reset_target")
def reset_target():
    if pipeline is not None:
        pipeline.target_tracker.reset()
        pipeline.raw_skeletons.clear()
        pipeline.normalized_skeletons.clear()
        return {"status": "success", "message": "Target lock reset."}
    return {"status": "idle", "message": "Pipeline is not initialized."}


if __name__ == "__main__":
    print("=============================================")
    print("Starting Web Realtime Server")
    print("Open http://localhost:8000 in your browser")
    print("=============================================")
    uvicorn.run(app, host="0.0.0.0", port=8000)
