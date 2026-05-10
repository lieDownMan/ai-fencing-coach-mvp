import gradio as gr
import json
import os
import pandas as pd
from pathlib import Path
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

def analyze_video(video_file, target_side, training_mode, user_str):
    if not video_file or not user_str:
        return None, "Please upload a video and select a user.", None
        
    user_id = get_user_choices().get(user_str)
    user = next((u for u in db.get_users() if u['id'] == user_id), None)
    
    pipeline = FullVideoPipeline(target_side=target_side, training_mode=training_mode)
    results = pipeline.process_video(video_file)
    
    annotator = VideoAnnotator()
    out_video = "annotated_output.mp4"
    if os.path.exists(out_video):
        os.remove(out_video)
    annotator.annotate_video(video_file, out_video, results)
    
    session_id = db.create_session(user_id, training_mode, out_video)
    db.save_action_logs(session_id, results["action_segments"], results["posture_errors"])
    
    summary = llm.generate_summary(user, training_mode, results["action_segments"], results["posture_errors"])
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
    
    return out_video, summary, table_df

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
                    video_input = gr.Video(label="Upload or Record Video", sources=["upload", "webcam"])
                    analyze_btn = gr.Button("Run Analysis", variant="primary")
                    
                with gr.Column(scale=2):
                    video_output = gr.Video(label="Processed Video")
                    summary_output = gr.Markdown(label="LLM Coach Summary", value="*Summary will appear here*")
                    action_table = gr.Dataframe(label="Drill-by-Drill Data")
                    
            analyze_btn.click(analyze_video, [video_input, target_side, training_mode, user_dropdown], [video_output, summary_output, action_table])
            
        with gr.Tab("History"):
            refresh_btn = gr.Button("Refresh History")
            history_table = gr.Dataframe(label="Past Sessions")
            refresh_btn.click(refresh_history, [], [history_table])
            app.load(refresh_history, [], [history_table])
            app.load(update_user_dropdown, [], [user_dropdown])

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860, share = True)
