"""
Main UI - Real-time dashboard for fencing coach system.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List
import logging

logger = logging.getLogger(__name__)


class FencingCoachUI:
    """
    Interactive dashboard for fencing coaching system.
    Displays live video, classifications, and coaching feedback.
    """
    
    def __init__(
        self,
        window_name: str = "AI Fencing Coach & Referee System",
        width: int = 1600,
        height: int = 900
    ):
        """
        Initialize UI.
        
        Args:
            window_name: OpenCV window title
            width: Display width
            height: Display height
        """
        self.window_name = window_name
        self.width = width
        self.height = height
        
        # Layout configuration
        self.video_panel_width = int(width * 0.6)
        self.video_panel_height = height
        
        self.info_panel_width = int(width * 0.4)
        self.info_panel_height = height
        
        # Display state
        self.current_frame = None
        self.current_classifications = []
        self.current_score = {"player": 0, "opponent": 0}
        self.current_feedback = ""
        self.action_frequencies = {}
        
        # Colors
        self.COLOR_BACKGROUND = (240, 240, 240)
        self.COLOR_TEXT = (0, 0, 0)
        self.COLOR_ACCENT = (255, 100, 0)  # Orange
        self.COLOR_POSITIVE = (0, 200, 0)  # Green
        self.COLOR_NEGATIVE = (0, 0, 255)  # Red
    
    def set_frame(self, frame: np.ndarray):
        """Set current video frame."""
        self.current_frame = frame.copy()
    
    def set_score(self, player_score: int, opponent_score: int):
        """Set current bout score."""
        self.current_score = {"player": player_score, "opponent": opponent_score}
    
    def set_classifications(self, classifications: List[str]):
        """Set recent classifications."""
        self.current_classifications = classifications[-10:]  # Last 10
    
    def set_action_frequencies(self, frequencies: Dict[str, float]):
        """Set action frequency distribution."""
        self.action_frequencies = frequencies
    
    def set_feedback(self, feedback: str):
        """Set coaching feedback text."""
        self.current_feedback = feedback
    
    def render(self) -> np.ndarray:
        """
        Render complete UI.
        
        Returns:
            Complete UI image (BGR)
        """
        # Create base canvas
        canvas = np.ones((self.height, self.width, 3), dtype=np.uint8) * 255
        
        # Render video panel (left)
        self._render_video_panel(canvas)
        
        # Render info panel (right)
        self._render_info_panel(canvas)
        
        return canvas
    
    def _render_video_panel(self, canvas: np.ndarray):
        """Render video display panel."""
        x_offset = 0
        y_offset = 0
        
        # Add title bar
        title_height = 40
        cv2.rectangle(
            canvas,
            (x_offset, y_offset),
            (x_offset + self.video_panel_width, y_offset + title_height),
            self.COLOR_ACCENT,
            -1
        )
        cv2.putText(
            canvas,
            "Live Video Feed",
            (x_offset + 10, y_offset + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2
        )
        
        # Add frame
        frame_y_start = y_offset + title_height
        frame_height = self.video_panel_height - title_height
        
        if self.current_frame is not None:
            # Resize frame to fit panel
            frame_resized = cv2.resize(
                self.current_frame,
                (self.video_panel_width, frame_height)
            )
            canvas[frame_y_start:frame_y_start + frame_height, 
                   x_offset:x_offset + self.video_panel_width] = frame_resized
    
    def _render_info_panel(self, canvas: np.ndarray):
        """Render information panel."""
        x_offset = self.video_panel_width
        y_offset = 0
        
        # Background
        cv2.rectangle(
            canvas,
            (x_offset, y_offset),
            (self.width, self.height),
            (245, 245, 245),
            -1
        )
        
        current_y = y_offset + 30
        line_height = 35
        
        # Title
        cv2.putText(
            canvas,
            "Coach's Corner",
            (x_offset + 15, current_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.1,
            self.COLOR_ACCENT,
            2
        )
        current_y += line_height + 10
        
        # Score display
        score_text = f"Score: {self.current_score['player']} - {self.current_score['opponent']}"
        cv2.putText(
            canvas,
            score_text,
            (x_offset + 15, current_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            self.COLOR_TEXT,
            2
        )
        current_y += line_height
        
        # Recent actions
        cv2.putText(
            canvas,
            "Recent Actions:",
            (x_offset + 15, current_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            self.COLOR_ACCENT,
            2
        )
        current_y += line_height
        
        # Display recent classifications
        actions_text = " -> ".join(self.current_classifications) if self.current_classifications else "None"
        # Wrap text if too long
        if len(actions_text) > 30:
            actions_text = actions_text[:30] + "..."
        
        cv2.putText(
            canvas,
            actions_text,
            (x_offset + 25, current_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            self.COLOR_TEXT,
            1
        )
        current_y += line_height + 10
        
        # Action frequencies
        cv2.putText(
            canvas,
            "Action Distribution:",
            (x_offset + 15, current_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            self.COLOR_ACCENT,
            2
        )
        current_y += line_height
        
        for action, freq in sorted(self.action_frequencies.items(), 
                                   key=lambda x: x[1], reverse=True)[:4]:
            freq_text = f"{action}: {freq:.0%}"
            cv2.putText(
                canvas,
                freq_text,
                (x_offset + 25, current_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                self.COLOR_TEXT,
                1
            )
            current_y += line_height - 10
        
        current_y += 10
        
        # Feedback section
        cv2.putText(
            canvas,
            "Coaching Feedback:",
            (x_offset + 15, current_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            self.COLOR_ACCENT,
            2
        )
        current_y += line_height
        
        # Render feedback box
        feedback_box_y = current_y
        feedback_box_height = self.height - feedback_box_y - 20
        
        cv2.rectangle(
            canvas,
            (x_offset + 10, feedback_box_y),
            (self.width - 10, feedback_box_y + feedback_box_height),
            self.COLOR_ACCENT,
            2
        )
        
        # Wrap and display feedback text
        feedback_lines = self._wrap_text(self.current_feedback, 35)
        feedback_y = feedback_box_y + 20
        
        for line in feedback_lines[:6]:  # Max 6 lines
            cv2.putText(
                canvas,
                line,
                (x_offset + 20, feedback_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                self.COLOR_TEXT,
                1
            )
            feedback_y += 25
    
    @staticmethod
    def _wrap_text(text: str, max_length: int) -> List[str]:
        """Wrap text to max length per line."""
        if not text:
            return []
        
        words = text.split()
        lines = []
        current_line = ""
        
        for word in words:
            if len(current_line) + len(word) + 1 <= max_length:
                current_line += word + " "
            else:
                if current_line:
                    lines.append(current_line.strip())
                current_line = word + " "
        
        if current_line:
            lines.append(current_line.strip())
        
        return lines
    
    def display(self, wait_key: int = 1):
        """Display UI window."""
        canvas = self.render()
        cv2.imshow(self.window_name, canvas)
        cv2.waitKey(wait_key)
    
    def close(self):
        """Close UI window."""
        cv2.destroyAllWindows()
    
    def save_screenshot(self, output_path: str):
        """Save current UI as image."""
        canvas = self.render()
        cv2.imwrite(output_path, canvas)
        logger.info(f"Screenshot saved: {output_path}")
