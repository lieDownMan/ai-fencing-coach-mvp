import gradio as gr
import cv2
import numpy as np
from realtime_app import LiveVideoPipeline

# 初始化我們剛剛寫好的即時 Pipeline
pipeline = LiveVideoPipeline()

def process_webcam_frame(frame):
    if frame is None:
        return None
        
    # Gradio 傳入的是 RGB 格式，轉成 BGR 讓 OpenCV 處理
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    # 進行即時骨架擷取、Sliding Window 推理與 Heuristics 判斷
    out_frame_bgr = pipeline.process_frame(frame_bgr)
    
    # 再轉回 RGB 讓網頁顯示
    out_frame_rgb = cv2.cvtColor(out_frame_bgr, cv2.COLOR_BGR2RGB)
    
    return out_frame_rgb

with gr.Blocks(title="Live Fencing Coach") as demo:
    gr.Markdown("# 🤺 AI Fencing Coach - Live Streaming")
    gr.Markdown("這個頁面會直接存取你的筆電鏡頭，並傳送到工作站進行即時分析。")
    
    with gr.Row():
        # 設定為 streaming=True，瀏覽器會不斷把鏡頭畫面送到後端
        input_video = gr.Image(sources=["webcam"], streaming=True, label="Live Webcam")
        output_video = gr.Image(label="Live Analysis Output")
        
    # 將輸入串流綁定到處理函數，並輸出到 output_video
    input_video.stream(fn=process_webcam_frame, inputs=[input_video], outputs=[output_video])

if __name__ == "__main__":
    # 使用 share=True 確保可以透過 HTTPS 繞過瀏覽器安全性限制
    demo.launch(server_name="0.0.0.0", server_port=7861, share=True)
