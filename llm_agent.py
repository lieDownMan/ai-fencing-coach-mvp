import os
from typing import Dict, Any, List
from dotenv import load_dotenv

load_dotenv()  # reads .env into os.environ

try:
    from google import genai
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

class LLMAgent:
    def __init__(self, api_key: str = None):
        key = api_key or os.environ.get("GEMINI_API_KEY")
        if HAS_GENAI and key:
            self.client = genai.Client(api_key=key)
            self.model_name = 'gemini-3.1-flash-lite-preview'
            self.enabled = True
        else:
            self.enabled = False
            self.disabled_reason = "GEMINI_API_KEY not set" if HAS_GENAI else "google-genai module not installed"
            
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
        
        prompt = f"""You are an elite, observant fencing coach. Your goal is to give a post-session summary based STRICTLY on objective biomechanical data extracted by our AI vision system. Do not invent errors or provide tactical advice that is not supported by the data.

[STUDENT PROFILE]
{user_info}

[SESSION CONTEXT]
Training Mode: {training_mode}
* Context Guide for Coach: 
  - If "Footwork", focus your advice on balance, center of mass stability, and stance width.
  - If "Target Practice", focus on kinetic chain (hand-before-foot), extension, and knee safety.
  - If "Free Bouting", focus on maintaining guard under pressure and action setup.

[OBJECTIVE ACTION STATS]
{stats_str}

[INSTRUCTIONS]
Based on the stats above, write a highly specific technical summary addressing the student directly (e.g., "Great job today...").
1. Acknowledge the volume/type of actions they practiced.
2. Identify the MOST frequent or critical biomechanical flaw listed in the stats.
3. Provide exactly ONE actionable, physical adjustment or mental cue to fix this specific flaw.
4. Tone: Direct, professional, and encouraging.
5. Constraint: Strictly under 100 words. Do NOT list timecodes. Please reply in Traditional Chinese.
"""
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
            )
            return response.text.strip()
        except Exception as e:
            return f"Error generating summary: {str(e)}"
