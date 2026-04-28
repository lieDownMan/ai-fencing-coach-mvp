import gradio as gr
import os
import json
import logging
from typing import Dict, Any
import pandas as pd

from src.app_interface.system_pipeline import SystemPipeline
from src.inference.video_annotator import VideoAnnotator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Preload pipeline placeholder
pipeline = None

def init_pipeline(target_side: str, weapon_hand: str):
    global pipeline
    logger.info("Initializing pipeline...")
    pipeline = SystemPipeline(
        use_bifencenet=False,
        model_checkpoint="weights/fencenet/best_model.pth", 
        target_side=target_side,
        weapon_hand=weapon_hand,
    )
    logger.info("Pipeline initialized.")

def analyze_video(video_file, target_side, weapon_hand, update_status):
    if video_file is None:
        return None, None, "Please upload a video file."

    # Update state via yield if stream/generator, but we just return text.
    logger.info(
        "Starting analysis for side %s with weapon hand %s",
        target_side,
        weapon_hand,
    )
    
    # 1. Initialize Pipeline with correct target_side
    init_pipeline(target_side, weapon_hand)
    
    # 2. Run Analysis (Modules 4 -> 5 -> 1 -> 2 -> 3)
    results = pipeline.process_video(video_path=video_file, fencer_id="player1")
    
    # Generate Posture Feedback
    feedback = pipeline.coach_engine.generate_posture_feedback(
        results.get("action_segments", []),
        results.get("posture_errors", [])
    )
    
    # 3. Create Annotated Video (Module 6)
    annotator = VideoAnnotator()
    output_video_path = "annotated_output.mp4"
    if os.path.exists(output_video_path):
        os.remove(output_video_path)
        
    annotator.annotate_video(video_file, output_video_path, results)
    
    # 4. Create Timeline Table (Module 6)
    source_fps = float(
        results.get("two_fencer_tracking", {}).get("source_fps") or 30.0
    )
    action_data = []
    for seg in results.get("action_segments", []):
        start_time = seg.get("video_start_frame", 0) / source_fps
        end_time = seg.get("video_end_frame", 0) / source_fps
        action_data.append([
            f"{start_time:.1f}s - {end_time:.1f}s",
            seg["action"],
            f"{seg['confidence']:.2f}"
        ])
    
    table_df = pd.DataFrame(action_data, columns=["Timeline", "Action", "Confidence"])
    
    # Include posture feedback in status
    status_text = f"Analysis Complete!\n\nAI Coach Feedback:\n{feedback}"
    
    return output_video_path, table_df, status_text

with gr.Blocks(title="AI Fencing Coach Validation UI") as app:
    gr.Markdown("# AI Fencing Coach Validation UI (Module 6 Pipeline)")
    
    with gr.Row():
        with gr.Column(scale=1):
            target_side_radio = gr.Radio(
                choices=["left", "right"], 
                value="right", 
                label="Target Fencer Side"
            )
            weapon_hand_radio = gr.Radio(
                choices=["auto", "left", "right"],
                value="auto",
                label="Weapon Hand",
                info="auto uses side-based inference; left/right forces the target fencer's weapon arm.",
            )
            video_input = gr.Video(label="Upload Raw Fencing Video")
            analyze_btn = gr.Button("Run Analysis", variant="primary")
            status_out = gr.Textbox(label="Status / Feedback", interactive=False, lines=5)
            
        with gr.Column(scale=2):
            video_output = gr.Video(label="Annotated Playback")
            action_table = gr.Dataframe(label="Action Timeline", headers=["Timeline", "Action", "Confidence"])

    
    analyze_btn.click(
        fn=analyze_video,
        inputs=[video_input, target_side_radio, weapon_hand_radio, status_out],
        outputs=[video_output, action_table, status_out]
    )

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860, share=True)
