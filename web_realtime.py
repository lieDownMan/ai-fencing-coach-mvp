import os
import cv2
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from pydantic import BaseModel
from src.realtime.realtime_app import LiveVideoPipeline

app = FastAPI()

# Config
CAMERA_SOURCE = "0"  # Can be 0, 1, or URL like "http://192.168.x.x:4747/video"
TARGET_SIDE = "left"
TRAINING_MODE = "Footwork"

pipeline = None
cap = None

class ConfigUpdate(BaseModel):
    camera_source: str
    target_side: str
    training_mode: str

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
            color: #ff3333;
            font-size: 2.2em;
            font-weight: bold;
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
        <button onclick="updateConfig()">Apply & Restart Camera</button>
    </div>

    <script>
        // Poll the server for pipeline status every 200ms
        setInterval(async () => {
            try {
                const response = await fetch('/status');
                const data = await response.json();
                
                document.getElementById('hud-state').innerText = `State: ${data.state}`;
                document.getElementById('hud-action').innerText = `Action: ${data.action}`;
                
                // Update warnings
                const warningsDiv = document.getElementById('hud-warnings');
                if (data.warnings && data.warnings.length > 0) {
                    warningsDiv.innerHTML = data.warnings.map(w => `⚠ ${w}`).join('<br>');
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
                
            } catch (err) {
                console.error("Error fetching status:", err);
            }
        }, 200);

        async function updateConfig() {
            const cam = document.getElementById('camera_source').value;
            const target = document.getElementById('target_side').value;
            const mode = document.getElementById('training_mode').value;

            try {
                const res = await fetch('/config', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        camera_source: cam,
                        target_side: target,
                        training_mode: mode
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
            blank = cv2.imencode('.jpg', cv2.resize(cv2.imread("web_outputs/placeholder.png") if os.path.exists("web_outputs/placeholder.png") else cv2.UMat(480, 640, cv2.CV_8UC3, (0, 0, 255))), [cv2.IMWRITE_JPEG_QUALITY, 50])[1].tobytes()
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


@app.get("/")
def index():
    html = HTML_PAGE.replace("{CAMERA_SOURCE}", CAMERA_SOURCE)
    
    html = html.replace("{TARGET_LEFT_SEL}", "selected" if TARGET_SIDE == "left" else "")
    html = html.replace("{TARGET_RIGHT_SEL}", "selected" if TARGET_SIDE == "right" else "")
    
    html = html.replace("{MODE_FW_SEL}", "selected" if TRAINING_MODE == "Footwork" else "")
    html = html.replace("{MODE_TP_SEL}", "selected" if TRAINING_MODE == "Target Practice" else "")
    html = html.replace("{MODE_FB_SEL}", "selected" if TRAINING_MODE == "Free Bouting" else "")

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
            "warnings": []
        })
        
    return JSONResponse({
        "state": getattr(pipeline.gatekeeper, "state", "UNKNOWN"),
        "action": pipeline.current_action,
        "warnings": pipeline.current_warnings if pipeline.warning_frames_left > 0 else []
    })


@app.post("/config")
def update_config(config: ConfigUpdate):
    """Endpoint to update global settings from the UI and reset the pipeline."""
    global CAMERA_SOURCE, TARGET_SIDE, TRAINING_MODE, pipeline, cap
    
    CAMERA_SOURCE = config.camera_source
    TARGET_SIDE = config.target_side
    TRAINING_MODE = config.training_mode
    
    # Release current resources to trigger a re-init
    if cap is not None:
        cap.release()
    cap = None
    
    if pipeline is not None and pipeline.voice_coach is not None:
        pipeline.voice_coach.shutdown()
    pipeline = None
    
    return {"status": "success", "message": "Configuration updated. Restarting pipeline."}


if __name__ == "__main__":
    print("=============================================")
    print("Starting Web Realtime Server")
    print("Open http://localhost:8000 in your browser")
    print("=============================================")
    uvicorn.run(app, host="0.0.0.0", port=8000)
