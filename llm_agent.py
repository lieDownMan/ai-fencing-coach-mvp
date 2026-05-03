import os
from typing import Dict, Any, List

try:
    import google.generativeai as genai
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

class LLMAgent:
    def __init__(self, api_key: str = None):
        key = api_key or os.environ.get("GEMINI_API_KEY")
        if HAS_GENAI and key:
            genai.configure(api_key=key)
            self.model = genai.GenerativeModel('gemini-1.5-flash')
            self.enabled = True
        else:
            self.enabled = False
            self.disabled_reason = "GEMINI_API_KEY not set" if HAS_GENAI else "google-generativeai module not installed"
            
    def generate_summary(self, user: Dict[str, Any], training_mode: str, action_segments: List[Dict], posture_errors: List[Dict]) -> str:
        if not self.enabled:
            return f"LLM Agent disabled: {self.disabled_reason}. Cannot generate summary."
            
        total_actions = len(action_segments)
        
        # Aggregate errors
        error_counts = {}
        for err in posture_errors:
            name = err.get("error", "Unknown error")
            error_counts[name] = error_counts.get(name, 0) + 1
            
        errors_str = ", ".join([f"{count} {name}" for name, count in error_counts.items()])
        if not errors_str:
            errors_str = "0 Errors"
            
        stats_str = f"Total actions: {total_actions}. Errors: {errors_str}"
        user_info = f"User is a {user.get('handedness', 'Right')}-handed fencer, height {user.get('height_cm', 180)}cm."
        
        prompt = f"""You are an expert fencing coach. Review the following session data for your student.
{user_info}
Training Mode: {training_mode}
Action Stats: {stats_str}

Write a short, encouraging, but highly specific technical paragraph (under 100 words) summarizing their performance and telling them exactly what biomechanical flaw to focus on fixing next. Do not list timecodes."""

        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            return f"Error generating summary: {str(e)}"
