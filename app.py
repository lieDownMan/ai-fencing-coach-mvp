import json
import os
import shutil
import pandas as pd
import socket
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent
_GRADIO_TEMP_DIR = _REPO_ROOT / "web_outputs" / "gradio_tmp"
_UPLOAD_DIR = _REPO_ROOT / "web_outputs" / "uploads"
_OUTPUT_DIR = _REPO_ROOT / "web_outputs" / "processed"
for _dir in (_GRADIO_TEMP_DIR, _UPLOAD_DIR, _OUTPUT_DIR):
    _dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("GRADIO_TEMP_DIR", str(_GRADIO_TEMP_DIR))

PROCESSING_PROFILES = {
    "Balanced": {
        "max_pose_width": 960,
        "pose_every_n_frames": 1,
        "annotated_max_width": 1280,
    },
    "Fast": {
        "max_pose_width": 640,
        "pose_every_n_frames": 2,
        "annotated_max_width": 960,
    },
    "Full Quality": {
        "max_pose_width": None,
        "pose_every_n_frames": 1,
        "annotated_max_width": None,
    },
}

import gradio as gr
import cv2
from inference.sliding_window import FullVideoPipeline
from inference.video_annotator import VideoAnnotator
from database import Database
from llm_agent import LLMAgent
from realtime_voice_coach import RealtimeVoiceCoach
import sqlite3

# Load playbook for error_key → error_name resolution in UI
_PLAYBOOK_PATH = Path(__file__).resolve().parent / "coach_playbook.json"
with open(_PLAYBOOK_PATH, "r", encoding="utf-8") as _f:
    _PLAYBOOK = json.load(_f)

def _resolve_error_name(error_key: str) -> str:
    entry = _PLAYBOOK.get(error_key)
    if entry:
        return entry.get("error_name", error_key)
    return error_key

db = Database()
llm = LLMAgent()
try:
    voice_coach = RealtimeVoiceCoach()
except Exception:
    voice_coach = None

# Create default user if none exists
if not db.get_users():
    db.create_user("Default User", "right", 180)

def get_user_choices():
    users = db.get_users()
    return {f"{u['name']} (ID: {u['id']})": u['id'] for u in users}

def update_user_dropdown():
    choices = list(get_user_choices().keys())
    return gr.update(choices=choices, value=choices[0] if choices else None)

def create_user_fn(name, handedness, height):
    db.create_user(name, handedness, int(height))
    return update_user_dropdown(), "User created successfully!"

def refresh_history():
    sessions = db.get_sessions()
    if not sessions:
        return pd.DataFrame(columns=["ID", "Date", "User", "Mode", "Summary"])
    data = []
    for s in sessions:
        data.append([
            s["session_id"], s["date"], s["user_name"], s["training_mode"], s["llm_summary"]
        ])
    return pd.DataFrame(data, columns=["ID", "Date", "User", "Mode", "Summary"])

def _video_path(video_file) -> Path:
    if isinstance(video_file, (str, Path)):
        return Path(video_file)
    if isinstance(video_file, dict):
        candidate = video_file.get("path") or video_file.get("name")
        if candidate:
            return Path(candidate)
    candidate = getattr(video_file, "path", None) or getattr(video_file, "name", None)
    if candidate:
        return Path(candidate)
    raise ValueError(f"Unsupported video input: {type(video_file).__name__}")

def _copy_video_to_workspace(video_file) -> str:
    source = _video_path(video_file)
    if not source.exists():
        raise FileNotFoundError(f"Uploaded video file not found: {source}")

    suffix = source.suffix if source.suffix else ".mp4"
    safe_stem = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in source.stem)
    target = _UPLOAD_DIR / f"{safe_stem}_{int(time.time() * 1000)}{suffix}"
    shutil.copy2(source, target)
    return str(target)

def _empty_action_table() -> pd.DataFrame:
    return pd.DataFrame(columns=["Start Time | End Time", "Action", "Warning"])

def _validate_video_file(video_path: str) -> dict:
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {path}")

    size_bytes = path.stat().st_size
    if size_bytes < 128:
        raise ValueError(
            f"Uploaded video is only {size_bytes} bytes. "
            "It looks incomplete or not fully saved yet. Try exporting/downloading the video again, then re-upload it."
        )

    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError(
                f"OpenCV could not open this video: {path.name}. "
                "The file may be corrupt, incomplete, or encoded in a format this OpenCV build cannot decode."
            )

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)

        ok, frame = cap.read()
        if not ok or frame is None:
            raise ValueError(
                f"OpenCV opened {path.name}, but could not read the first frame. "
                "This usually means the upload is incomplete or the video metadata is damaged."
            )

        if width <= 0 or height <= 0:
            height, width = frame.shape[:2]
        if frame_count <= 0:
            frame_count = 1
        if fps <= 0:
            fps = 30.0
    finally:
        cap.release()

    return {
        "size_bytes": size_bytes,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "fps": fps,
    }

def analyze_video(
    video_file,
    target_side,
    training_mode,
    processing_profile,
    use_llm_summary,
    user_str,
    progress=gr.Progress(),
):
    if not video_file or not user_str:
        return None, "Please upload a video and select a user.", _empty_action_table()

    profile = PROCESSING_PROFILES.get(processing_profile, PROCESSING_PROFILES["Balanced"])
    progress(0.02, desc="Copying upload")
    user_id = get_user_choices().get(user_str)
    user = next((u for u in db.get_users() if u['id'] == user_id), None)
    try:
        input_video_path = _copy_video_to_workspace(video_file)
        video_info = _validate_video_file(input_video_path)
    except (FileNotFoundError, ValueError) as exc:
        progress(1.0, desc="Stopped")
        return None, f"Video upload problem: {exc}", _empty_action_table()
    progress(
        0.04,
        desc=f"Validated {video_info['width']}x{video_info['height']} video",
    )

    pipeline = FullVideoPipeline(
        target_side=target_side,
        training_mode=training_mode,
        max_pose_width=profile["max_pose_width"],
        pose_every_n_frames=profile["pose_every_n_frames"],
    )
    results = pipeline.process_video(
        input_video_path,
        progress_callback=lambda fraction, desc: progress(fraction, desc=desc),
    )

    annotator = VideoAnnotator()
    out_video = str(_OUTPUT_DIR / f"annotated_{int(time.time() * 1000)}.mp4")
    if os.path.exists(out_video):
        os.remove(out_video)
    progress(0.9, desc="Rendering annotated video")
    annotator.annotate_video(
        input_video_path,
        out_video,
        results,
        max_width=profile["annotated_max_width"],
    )

    session_id = db.create_session(user_id, training_mode, out_video)
    db.save_action_logs(session_id, results["action_segments"], results["posture_errors"])

    progress(0.96, desc="Writing session summary")
    summary = llm.generate_summary(
        user,
        training_mode,
        results["action_segments"],
        results["posture_errors"],
        use_llm=use_llm_summary,
    )
    db.update_session_summary(session_id, summary)

    action_data = []
    for seg in results.get("action_segments", []):
        start_time = seg.get("video_start_frame", 0) / 30.0
        end_time = seg.get("video_end_frame", 0) / 30.0

        # Find warnings in this segment
        warning = ""
        for err in results.get("posture_errors", []):
            if err.get("start_frame", 0) >= seg.get("video_start_frame", 0) and err.get("start_frame", 0) <= seg.get("video_end_frame", 0):
                error_key = err.get("error_key", err.get("error", ""))
                warning = _resolve_error_name(error_key)
                # Fire real-time voice cue (non-blocking)
                if voice_coach and error_key:
                    voice_coach.speak_async(error_key)
                break

        action_data.append([
            f"{start_time:.1f}s - {end_time:.1f}s",
            seg["action"],
            warning
        ])

    table_df = pd.DataFrame(action_data, columns=["Start Time | End Time", "Action", "Warning"])

    progress(1.0, desc="Done")
    return out_video, summary, table_df

def _pick_gradio_port(default_port: int) -> int:
    env_port = os.getenv("GRADIO_SERVER_PORT")
    if env_port:
        return int(env_port)

    for port in range(default_port, default_port + 20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.2)
            if sock.connect_ex(("127.0.0.1", port)) != 0:
                return port
    return default_port

with gr.Blocks(title="AI Fencing Coach") as app:
    gr.Markdown("# AI Fencing Coach")
    
    with gr.Tabs():
        with gr.Tab("Analysis"):
            with gr.Row():
                with gr.Column(scale=1):
                    user_dropdown = gr.Dropdown(choices=list(get_user_choices().keys()), label="Select User")
                    with gr.Accordion("Create New User", open=False):
                        new_name = gr.Textbox(label="Name")
                        new_hand = gr.Radio(["left", "right"], label="Handedness", value="right")
                        new_height = gr.Number(label="Height (cm)", value=180)
                        create_btn = gr.Button("Create")
                        create_msg = gr.Textbox(show_label=False, interactive=False)
                        create_btn.click(create_user_fn, [new_name, new_hand, new_height], [user_dropdown, create_msg])
                        
                    target_side = gr.Radio(["left", "right"], value="left", label="Target Fencer")
                    training_mode = gr.Radio(["Footwork", "Target Practice", "Free Bouting"], value="Footwork", label="Training Mode")
                    processing_profile = gr.Radio(
                        list(PROCESSING_PROFILES.keys()),
                        value="Balanced",
                        label="Processing",
                    )
                    use_llm_summary = gr.Checkbox(
                        label="Use Gemini Summary",
                        value=llm.enabled,
                        interactive=llm.enabled,
                    )
                    video_input = gr.File(
                        label="Upload Video",
                        file_types=[".mp4", ".mov", ".avi", ".mkv"],
                        type="filepath",
                    )
                    analyze_btn = gr.Button("Run Analysis", variant="primary")
                    
                with gr.Column(scale=2):
                    video_output = gr.Video(label="Processed Video")
                    summary_output = gr.Markdown(label="LLM Coach Summary", value="*Summary will appear here*")
                    action_table = gr.Dataframe(label="Drill-by-Drill Data")
                    
            analyze_btn.click(
                analyze_video,
                [
                    video_input,
                    target_side,
                    training_mode,
                    processing_profile,
                    use_llm_summary,
                    user_dropdown,
                ],
                [video_output, summary_output, action_table],
            )
            
        with gr.Tab("History"):
            refresh_btn = gr.Button("Refresh History")
            history_table = gr.Dataframe(label="Past Sessions")
            refresh_btn.click(refresh_history, [], [history_table])
            app.load(refresh_history, [], [history_table])
            app.load(update_user_dropdown, [], [user_dropdown])

if __name__ == "__main__":
    port = _pick_gradio_port(7860)
    print(f"Launching AI Fencing Coach at http://127.0.0.1:{port}")
    app.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=os.getenv("GRADIO_SHARE", "0") == "1",
        allowed_paths=[str(_REPO_ROOT / "web_outputs")],
    )
