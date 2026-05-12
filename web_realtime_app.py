import os
import socket

import cv2
import gradio as gr
from realtime_app import LiveVideoPipeline


pipeline = LiveVideoPipeline()


def process_webcam_frame(frame):
    if frame is None:
        return None

    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    out_frame_bgr = pipeline.process_frame(frame_bgr)
    return cv2.cvtColor(out_frame_bgr, cv2.COLOR_BGR2RGB)


with gr.Blocks(title="Live Fencing Coach") as demo:
    gr.Markdown("# AI Fencing Coach - Live Streaming")
    gr.Markdown("Use the browser webcam to stream frames through the live coaching pipeline.")

    with gr.Row():
        input_video = gr.Image(sources=["webcam"], streaming=True, label="Live Webcam")
        output_video = gr.Image(label="Live Analysis Output")

    input_video.stream(fn=process_webcam_frame, inputs=[input_video], outputs=[output_video])


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


if __name__ == "__main__":
    port = _pick_gradio_port(7861)
    print(f"Launching Live Fencing Coach at http://127.0.0.1:{port}")
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=os.getenv("GRADIO_SHARE", "0") == "1",
    )
