import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()  # reads .env into os.environ

_PLAYBOOK_PATH = Path(__file__).resolve().parent / "coach_playbook.json"
try:
    with open(_PLAYBOOK_PATH, "r", encoding="utf-8") as _f:
        _PLAYBOOK = json.load(_f)
except FileNotFoundError:
    _PLAYBOOK = {}

try:
    # Use google-generativeai (legacy SDK) to avoid websockets conflict with gradio on Python 3.9
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
            self.disabled_reason = (
                "GEMINI_API_KEY not set"
                if HAS_GENAI
                else "google-generativeai module not installed"
            )

    @staticmethod
    def _error_key(err: Dict[str, Any]) -> str:
        return err.get("error_key") or err.get("error") or "unknown_error"

    def _aggregate_playbook_errors(
        self,
        posture_errors: List[Dict],
        focus_errors: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        focus_set = set(focus_errors or [])
        counts = Counter(self._error_key(err) for err in posture_errors)
        items = []
        for key, count in counts.items():
            entry = _PLAYBOOK.get(key, {})
            items.append(
                {
                    "key": key,
                    "count": count,
                    "error_name": entry.get("error_name", key),
                    "diagnosis": entry.get("diagnosis", ""),
                    "short_cue": entry.get("short_cue", ""),
                    "focused": key in focus_set,
                }
            )
        return sorted(
            items,
            key=lambda item: (
                not item["focused"],
                -item["count"],
                item["error_name"],
            ),
        )

    @staticmethod
    def _format_playbook_block(playbook_errors: List[Dict[str, Any]]) -> str:
        if not playbook_errors:
            return "No posture problems were detected."

        lines = []
        for item in playbook_errors:
            lines.append(
                "\n".join(
                    [
                        f"- error_key: {item['key']}",
                        f"  frequency: {item['count']}",
                        f"  problem: {item['error_name']}",
                        (
                            "  diagnosis: "
                            f"{item['diagnosis'] or 'No playbook diagnosis available.'}"
                        ),
                        (
                            "  short_cue: "
                            f"{item['short_cue'] or 'No playbook cue available.'}"
                        ),
                    ]
                )
            )
        return "\n".join(lines)

    @staticmethod
    def _generate_rule_based_summary(
        training_mode: str,
        total_actions: int,
        playbook_errors: List[Dict[str, Any]],
    ) -> str:
        lines = [f"本次 {training_mode} 共辨識到 {total_actions} 個動作。"]
        if not playbook_errors:
            lines.append("未偵測到 coach_playbook.json 中定義的姿勢問題。")
            return "\n".join(lines)

        lines.append("偵測到的問題與頻率：")
        for item in playbook_errors:
            detail = f"- {item['error_name']}：{item['count']} 次"
            if item["diagnosis"]:
                detail += f"。{item['diagnosis']}"
            if item["short_cue"]:
                detail += f" 教練提示：{item['short_cue']}"
            lines.append(detail)
        return "\n".join(lines)

    def generate_summary(
        self,
        user: Dict[str, Any],
        training_mode: str,
        action_segments: List[Dict],
        posture_errors: List[Dict],
        use_llm: bool = True,
        focus_errors: Optional[List[str]] = None,
    ) -> str:
        user = user or {}
        total_actions = len(action_segments)
        playbook_errors = self._aggregate_playbook_errors(posture_errors, focus_errors)

        if not use_llm or not self.enabled:
            return self._generate_rule_based_summary(
                training_mode, total_actions, playbook_errors
            )

        errors_str = (
            ", ".join(
                f"{item['count']}x {item['error_name']}"
                for item in playbook_errors
            )
            or "0 Errors"
        )
        stats_str = f"Total actions: {total_actions}. Errors: {errors_str}"
        user_info = (
            f"User is a {user.get('handedness', 'Right')}-handed fencer, "
            f"height {user.get('height_cm', 180)}cm."
        )
        playbook_block = self._format_playbook_block(playbook_errors)
        focus_line = ""
        if focus_errors:
            focused_names = []
            for key in focus_errors:
                entry = _PLAYBOOK.get(key, {})
                focused_names.append(entry.get("error_name", key))
            focus_line = (
                "\n[USER FEEDBACK FOCUS]\n"
                "The user chose to prioritize these problems in this review: "
                f"{', '.join(focused_names)}.\n"
            )

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

[COACH PLAYBOOK CONTEXT]
The following detected problems come from coach_playbook.json. Treat this as the source of truth for problem names, diagnoses, cue wording, and frequency:
{playbook_block}
{focus_line}

[INSTRUCTIONS]
Based on the stats and coach playbook context above, write a highly specific technical summary addressing the student directly.
1. Acknowledge the volume/type of actions they practiced.
2. List every detected problem and how many times it appeared.
3. Use the playbook diagnosis and short cue when explaining each problem.
4. If multiple problems were detected, prioritize the most frequent one at the end.
5. Tone: Direct, professional, and encouraging.
6. Constraint: Strictly under 160 words. Do NOT list timecodes. Please reply in Traditional Chinese.
"""
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            fallback = self._generate_rule_based_summary(
                training_mode, total_actions, playbook_errors
            )
            return f"Gemini summary failed: {str(e)}\n\n{fallback}"
