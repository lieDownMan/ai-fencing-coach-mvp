"""
Video Annotator
Spec reference: fixing_app.md § Module 6

Provides video burn-in for bounding boxes, skeleton, and HUD.
"""

import cv2
import numpy as np
import logging
from typing import Dict, Any, List, Optional
import os

logger = logging.getLogger(__name__)

class VideoAnnotator:
    # Colors (BGR)
    COLOR_BBOX = (0, 255, 0)
    COLOR_SKEL = (0, 255, 255)
    COLOR_HUD_BG = (50, 50, 50)
    COLOR_TEXT = (255, 255, 255)
    
    # Body connections
    CONNECTIONS = [
        ("nose", "front_shoulder"),
        ("front_shoulder", "front_elbow"),
        ("front_elbow", "front_wrist"),
        ("front_shoulder", "left_hip"),
        ("front_shoulder", "right_hip"),
        ("left_hip", "right_hip"),
        ("left_hip", "left_knee"),
        ("left_knee", "left_ankle"),
        ("right_hip", "right_knee"),
        ("right_knee", "right_ankle")
    ]

    @staticmethod
    def _frames_by_index(frames_meta: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
        """Map sparse tracking metadata to actual video frame indices."""
        return {
            int(frame.get("frame_index", index)): frame
            for index, frame in enumerate(frames_meta or [])
        }

    @staticmethod
    def _select_target_track(
        frame_info: Dict[str, Any],
        target_side: Optional[str],
    ) -> Optional[Dict[str, Any]]:
        """Return the tracked fencer matching the requested side, or a safe fallback."""
        tracks = frame_info.get("tracks", [])
        if target_side:
            for track in tracks:
                if track.get("side") == target_side:
                    return track
        if tracks:
            return max(tracks, key=lambda track: float(track.get("area", 0.0)))
        return None

    def annotate_video(self, input_path: str, output_path: str, report: Dict[str, Any]) -> str:
        """
        Burn annotations into the video and save to output_path.
        """
        tracking = report.get("two_fencer_tracking", {})
        frames_meta = tracking.get("frames", [])
        target_side = tracking.get("target_side")
        action_segments = report.get("action_segments", [])
        frames_by_index = self._frames_by_index(frames_meta)
        
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {input_path}")
            
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not out.isOpened():
            cap.release()
            raise ValueError(f"Cannot open VideoWriter for {output_path}")

        frame_idx = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_info = frames_by_index.get(frame_idx)
            
            # Find current action
            current_action = "None"
            current_conf = 0.0
            for seg in action_segments:
                if seg.get("video_start_frame", 0) <= frame_idx <= seg.get("video_end_frame", 0):
                    current_action = seg["action"]
                    current_conf = seg["confidence"]
                    break
                    
            if frame_info:
                # 1. Draw target bounding box and skeleton
                target_det = self._select_target_track(
                    frame_info=frame_info,
                    target_side=target_side,
                )
                            
                if target_det:
                    # Bounding Box
                    bbox = target_det.get("bbox")
                    if bbox and len(bbox) == 4:
                        x1, y1, x2, y2 = [int(v) for v in bbox]
                        cv2.rectangle(frame, (x1, y1), (x2, y2), self.COLOR_BBOX, 2)
                        
                    # Skeleton
                    skel = target_det.get("skeleton")
                    if skel:
                        # Draw joints
                        for pt in skel.values():
                            cv2.circle(frame, (int(pt[0]), int(pt[1])), 4, self.COLOR_SKEL, -1)
                        # Draw lines
                        for joint1, joint2 in self.CONNECTIONS:
                            if joint1 in skel and joint2 in skel:
                                pt1 = (int(skel[joint1][0]), int(skel[joint1][1]))
                                pt2 = (int(skel[joint2][0]), int(skel[joint2][1]))
                                cv2.line(frame, pt1, pt2, self.COLOR_SKEL, 2)
                
                # 2. Draw HUD
                state = frame_info.get("gatekeeper_state", "UNKNOWN")
                knee_angle = frame_info.get("knee_angle", 180.0)
                
                hud_text = [
                    f"State: {state}",
                    f"Action: {current_action} ({current_conf:.2f})",
                    f"Knee Angle: {knee_angle:.1f}"
                ]
                
                # Draw semi-transparent background
                overlay = frame.copy()
                cv2.rectangle(overlay, (10, 10), (350, 120), self.COLOR_HUD_BG, -1)
                cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
                
                # Draw text
                y_offset = 40
                for text in hud_text:
                    if "State: ACTIVE" in text:
                        color = (0, 255, 0)
                    elif "State: IDLE" in text:
                        color = (0, 0, 255)
                    else:
                        color = self.COLOR_TEXT
                        
                    cv2.putText(frame, text, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    y_offset += 35
                    
            out.write(frame)
            frame_idx += 1
            
        cap.release()
        out.release()
        logger.info(f"Saved annotated video to {output_path}")
        return output_path
