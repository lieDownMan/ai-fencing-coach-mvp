import cv2
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class VideoAnnotator:
    COLOR_BBOX_GREEN = (0, 255, 0)
    COLOR_BBOX_RED = (0, 0, 255)
    COLOR_SKEL = (0, 255, 255)
    COLOR_HUD_BG = (50, 50, 50)
    COLOR_TEXT = (255, 255, 255)
    
    CONNECTIONS = [
        ("nose", "front_shoulder"), ("front_shoulder", "front_elbow"),
        ("front_elbow", "front_wrist"), ("front_shoulder", "left_hip"),
        ("front_shoulder", "right_hip"), ("left_hip", "right_hip"),
        ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
        ("right_hip", "right_knee"), ("right_knee", "right_ankle")
    ]

    def annotate_video(self, input_path: str, output_path: str, report: Dict[str, Any]) -> str:
        tracking = report.get("two_fencer_tracking", {})
        frames_meta = tracking.get("frames", [])
        locked_track_id = tracking.get("locked_track_id", None)
        action_segments = report.get("action_segments", [])
        posture_errors = report.get("posture_errors", [])
        training_mode = report.get("training_mode", "Free Bouting")
        
        frames_dict = {f.get("frame_index"): f for f in frames_meta}
        
        cap = cv2.VideoCapture(input_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret: break
                
            frame_info = frames_dict.get(frame_idx)
            current_action = "None"
            current_alert = ""
            
            for seg in action_segments:
                if seg.get("video_start_frame", 0) <= frame_idx <= seg.get("video_end_frame", 0):
                    current_action = seg["action"]
                    break
                    
            for err in posture_errors:
                if err.get("start_frame", 0) <= frame_idx <= err.get("end_frame", 0):
                    current_alert = err.get("error", "")
                    break
                    
            if frame_info:
                target_det = None
                if locked_track_id is not None:
                    for det in frame_info.get("tracks", []):
                        if det.get("track_id") == locked_track_id:
                            target_det = det
                            break
                            
                if target_det:
                    bbox = target_det.get("bbox")
                    if bbox and len(bbox) == 4:
                        x1, y1, x2, y2 = [int(v) for v in bbox]
                        color = self.COLOR_BBOX_RED if current_alert else self.COLOR_BBOX_GREEN
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4)
                        
                    skel = target_det.get("skeleton")
                    if skel:
                        for pt in skel.values():
                            cv2.circle(frame, (int(pt[0]), int(pt[1])), 4, self.COLOR_SKEL, -1)
                        for joint1, joint2 in self.CONNECTIONS:
                            if joint1 in skel and joint2 in skel:
                                pt1 = (int(skel[joint1][0]), int(skel[joint1][1]))
                                pt2 = (int(skel[joint2][0]), int(skel[joint2][1]))
                                cv2.line(frame, pt1, pt2, self.COLOR_SKEL, 2)
                
                state = frame_info.get("gatekeeper_state", "UNKNOWN")
                
                hud_text = [
                    f"Mode: {training_mode}",
                    f"State: {state}",
                    f"Action: {current_action}"
                ]
                if current_alert:
                    hud_text.append(f"Alert: {current_alert}")
                
                overlay = frame.copy()
                cv2.rectangle(overlay, (10, 10), (600, 150 if current_alert else 120), self.COLOR_HUD_BG, -1)
                cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
                
                y_offset = 40
                for text in hud_text:
                    color = self.COLOR_TEXT
                    if text.startswith("Alert:"): color = (0, 0, 255) # Red
                    elif "State: ACTIVE" in text: color = (0, 255, 0) # Green
                    elif "State: IDLE" in text: color = (0, 0, 255) # Red
                    
                    cv2.putText(frame, text, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    y_offset += 35
                    
            out.write(frame)
            frame_idx += 1
            
        cap.release()
        out.release()
        return output_path
