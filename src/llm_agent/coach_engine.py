"""
Coach Engine - Orchestrates coaching feedback generation using LLM + tracking data.
"""

from typing import Dict, List, Optional, Any
import logging

from .prompt_templates import PromptTemplates
from .model_loader import ModelLoader
from ..tracking import PatternAnalyzer, ProfileManager

logger = logging.getLogger(__name__)


class CoachEngine:
    """
    Virtual coaching engine that combines LLM inference with fencing pattern analysis.
    Generates contextual coaching feedback based on real-time performance data.
    """
    
    def __init__(
        self,
        model_name: str = "llava-next",
        profiles_dir: str = "data/fencer_profiles/",
        device: str = "auto"
    ):
        """
        Initialize Coach Engine.
        
        Args:
            model_name: Name of LLM model to load
            profiles_dir: Directory containing fencer profiles
            device: Device to load model on
        """
        self.model_loader = ModelLoader(model_name=model_name, device=device)
        self.profile_manager = ProfileManager(profiles_dir=profiles_dir)
        self.pattern_analyzer = PatternAnalyzer()
        
        try:
            model_loaded = self.model_loader.load_model()
            if not model_loaded:
                logger.info("Coach engine will operate in analysis-only mode")
        except Exception as e:
            logger.warning(f"Could not load LLM model: {e}")
            logger.info("Coach engine will operate in analysis-only mode")
    
    def generate_immediate_feedback(
        self,
        fencer_id: str,
        current_score: Dict[str, int],
        num_recent_actions: int = 5,
        stats: Optional[Dict[str, Any]] = None,
        recent_actions: Optional[List[str]] = None
    ) -> str:
        """
        Generate immediate feedback during bout.
        
        Args:
            fencer_id: Fencer identifier
            current_score: Current bout score
            num_recent_actions: Number of recent actions to consider
            stats: Optional statistics from the active pipeline analyzer
            recent_actions: Optional recent action labels from the active bout
            
        Returns:
            Feedback text
        """
        logger.info(f"Generating immediate feedback for {fencer_id}")
        
        stats = stats or self.pattern_analyzer.get_statistics_summary()
        action_freqs = stats.get("action_frequencies", {}) or {}
        offensive_ratio = stats.get("offensive_ratio", 0.0)
        defensive_ratio = stats.get("defensive_ratio", 0.0)
        patterns = stats.get("repetitive_patterns", []) or []
        
        if recent_actions is None:
            recent_actions = self.pattern_analyzer.action_history
        recent_actions = list(recent_actions[-num_recent_actions:])
        
        # Generate prompt
        prompt = PromptTemplates.get_immediate_feedback_prompt(
            action_frequencies=action_freqs,
            offensive_ratio=offensive_ratio,
            defensive_ratio=defensive_ratio,
            recent_actions=recent_actions,
            current_score=current_score,
            patterns=patterns
        )
        
        feedback = self._try_generate(
            prompt=prompt,
            max_new_tokens=100,
            temperature=0.7
        )
        if feedback is None:
            feedback = self._generate_analytical_feedback(
                action_freqs, offensive_ratio, defensive_ratio
            )
        
        return feedback
    
    def generate_break_strategy(
        self,
        fencer_id: str,
        opponent_id: Optional[str] = None,
        current_score: Optional[Dict[str, int]] = None,
        fencer_stats: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate strategic advice during bout break.
        
        Args:
            fencer_id: Fencer identifier
            opponent_id: Optional opponent identifier
            current_score: Current bout score
            fencer_stats: Optional statistics from the active pipeline analyzer
            
        Returns:
            Strategy text
        """
        logger.info(f"Generating break strategy for {fencer_id}")
        
        current_score = current_score or {"player": 0, "opponent": 0}
        
        fencer_stats = self._with_most_frequent_action(
            fencer_stats or self.pattern_analyzer.get_statistics_summary()
        )
        js_sf_ratio = fencer_stats.get("js_sf_ratio", 0.0)
        
        # Try to get opponent patterns
        opponent_patterns = {}
        if opponent_id:
            try:
                opponent_profile = self.profile_manager.load_profile(opponent_id)
                if opponent_profile:
                    opponent_patterns = self._extract_opponent_patterns(opponent_profile)
            except Exception as e:
                logger.warning(f"Could not load opponent profile: {e}")
        
        # Generate prompt
        prompt = PromptTemplates.get_break_strategy_prompt(
            fencer_stats=fencer_stats,
            opponent_patterns=opponent_patterns,
            js_sf_ratio=js_sf_ratio,
            current_score=current_score
        )
        
        strategy = self._try_generate(
            prompt=prompt,
            max_new_tokens=150,
            temperature=0.7
        )
        if strategy is None:
            strategy = self._generate_analytical_strategy(fencer_stats, opponent_patterns)
        
        return strategy
    
    def generate_conclusive_feedback(
        self,
        fencer_id: str,
        bout_result: str,
        bout_statistics: Dict[str, Any],
        opponent_id: Optional[str] = None
    ) -> str:
        """
        Generate post-bout conclusive feedback.
        
        Args:
            fencer_id: Fencer identifier
            bout_result: Bout result (win/loss/draw)
            bout_statistics: Statistics from the bout
            opponent_id: Optional opponent identifier
            
        Returns:
            Conclusive feedback text
        """
        logger.info(f"Generating conclusive feedback for {fencer_id}")
        
        # Load fencer profile and get historical data
        profile = self.profile_manager.load_profile(fencer_id)
        
        historical_data = {}
        if profile:
            overall_stats = profile.get("overall_stats", {})
            historical_data = {
                "avg_offensive": overall_stats.get("average_offensive_ratio", 0.0),
                "avg_defensive": overall_stats.get("average_defensive_ratio", 0.0),
                "win_rate": (
                    overall_stats.get("wins", 0) / 
                    max(overall_stats.get("total_bouts", 1), 1)
                ),
                "trends": self._get_performance_trends(profile)
            }
        
        # Opponent info
        opponent_info = {}
        if opponent_id:
            opponent_profile = self.profile_manager.load_profile(opponent_id)
            if opponent_profile:
                opponent_stats = opponent_profile.get("overall_stats", {})
                opponent_info = {
                    "level": self._estimate_opponent_level(opponent_stats),
                    "win_rate": opponent_stats.get("wins", 0) / 
                               max(opponent_stats.get("total_bouts", 1), 1)
                }
        
        # Generate prompt
        prompt = PromptTemplates.get_conclusive_feedback_prompt(
            bout_stats=bout_statistics,
            historical_progression=historical_data,
            bout_result=bout_result,
            opponent_info=opponent_info
        )
        
        feedback = self._try_generate(
            prompt=prompt,
            max_new_tokens=300,
            temperature=0.7
        )
        if feedback is None:
            feedback = self._generate_analytical_conclusive(
                bout_statistics, historical_data, bout_result
            )
        
        return feedback

    def _try_generate(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float
    ) -> Optional[str]:
        """Generate with LLM when available; otherwise let callers use fallback."""
        if not self.model_loader.is_loaded():
            return None

        try:
            response = self.model_loader.generate(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature
            )
        except Exception as e:
            logger.warning(f"LLM generation failed; using analytical fallback: {e}")
            return None

        response = response.strip()
        return response or None

    def _with_most_frequent_action(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure prompt stats include a readable most-frequent action field."""
        enriched_stats = dict(stats)
        action_freqs = enriched_stats.get("action_frequencies", {}) or {}
        active_actions = [
            (action, freq)
            for action, freq in action_freqs.items()
            if freq > 0
        ]
        if active_actions and "most_frequent_action" not in enriched_stats:
            most_frequent_action, _ = max(active_actions, key=lambda item: item[1])
            enriched_stats["most_frequent_action"] = most_frequent_action

        return enriched_stats
    
    def _generate_analytical_feedback(
        self,
        action_freqs: Dict[str, float],
        offensive_ratio: float,
        defensive_ratio: float
    ) -> str:
        """Generate feedback based on analysis (LLM fallback)."""
        top_actions = [
            (action, freq)
            for action, freq in sorted(
                action_freqs.items(),
                key=lambda item: item[1],
                reverse=True
            )
            if freq > 0
        ][:3]
        actions_str = ", ".join(
            f"{action}: {freq:.0%}"
            for action, freq in top_actions
        )

        if not actions_str:
            return "Focus on execution. No stable action pattern yet; keep your distance and build clean entries."

        feedback = f"Focus on execution. Your primary actions: {actions_str}. "
        if offensive_ratio > 0.6:
            feedback += "Maintain aggressive momentum."
        elif defensive_ratio > 0.4:
            feedback += "Look for offensive opportunities."
        else:
            feedback += "Keep varying your entries and watch the opponent's response."
        
        return feedback
    
    def _generate_analytical_strategy(
        self,
        fencer_stats: Dict[str, Any],
        opponent_patterns: Dict[str, Any]
    ) -> str:
        """Generate strategy based on analysis (LLM fallback)."""
        strategy = "Strategic Adjustment: "
        if fencer_stats.get("total_actions", 0) == 0:
            return strategy + "collect a few exchanges first, then adjust based on real action patterns."
        
        if opponent_patterns.get("defensive_ratio", 0) > 0.5:
            strategy += "Opponent is defensive - use more offensive actions. "
        else:
            strategy += "Opponent is aggressive - be prepared to defend. "
        
        js_sf = fencer_stats.get("js_sf_ratio", 0)
        if js_sf < 0.5:
            strategy += "Increase variety in footwork patterns."
        
        return strategy
    
    def _generate_analytical_conclusive(
        self,
        bout_stats: Dict[str, Any],
        historical_data: Dict[str, Any],
        bout_result: str
    ) -> str:
        """Generate conclusive feedback based on analysis (LLM fallback)."""
        feedback = f"Bout Analysis - Result: {bout_result.upper()}\n"
        feedback += f"Performance: Offensive {bout_stats.get('offensive_ratio', 0):.0%} | "
        feedback += f"Defensive {bout_stats.get('defensive_ratio', 0):.0%}\n"
        
        if historical_data.get("win_rate", 0) > 0.6:
            feedback += "Strong overall performance trend. Keep up the momentum!"
        else:
            feedback += "Focus on technical consistency in next session."
        
        return feedback
    
    def _extract_opponent_patterns(self, opponent_profile: Dict) -> Dict:
        """Extract opponent tactical patterns from profile."""
        bouts = opponent_profile.get("bouts", [])
        
        if not bouts:
            return {}
        
        # Analyze recent bouts
        recent_bouts = bouts[-5:]  # Last 5 bouts
        
        defensive_ratios = [
            bout.get("statistics", {}).get("defensive_ratio", 0)
            for bout in recent_bouts
        ]
        
        patterns = {
            "defensive_ratio": sum(defensive_ratios) / len(defensive_ratios) if defensive_ratios else 0,
            "patterns": opponent_profile.get("overall_stats", {}).get("average_js_sf_ratio", 0)
        }
        
        return patterns
    
    def _get_performance_trends(self, profile: Dict) -> List[str]:
        """Extract performance trends from fencer profile."""
        bouts = profile.get("bouts", [])
        
        if len(bouts) < 2:
            return ["Insufficient history"]
        
        # Compare recent performance to older performance
        recent = bouts[-3:]
        older = bouts[:-3]
        
        recent_offensive = sum(
            b.get("statistics", {}).get("offensive_ratio", 0) for b in recent
        ) / len(recent) if recent else 0
        
        older_offensive = sum(
            b.get("statistics", {}).get("offensive_ratio", 0) for b in older
        ) / len(older) if older else 0
        
        trends = []
        if recent_offensive > older_offensive:
            trends.append("Increasing offensive aggression")
        if recent_offensive < older_offensive:
            trends.append("More conservative approach")
        
        return trends if trends else ["Stable performance"]
    
    def _estimate_opponent_level(self, opponent_stats: Dict) -> str:
        """Estimate opponent skill level."""
        win_rate = (
            opponent_stats.get("wins", 0) /
            max(opponent_stats.get("total_bouts", 1), 1)
        )
        
        if win_rate > 0.7:
            return "Elite"
        elif win_rate > 0.5:
            return "Advanced"
        else:
            return "Intermediate"

    def generate_posture_feedback(
        self,
        action_segments: List[Dict[str, Any]],
        posture_errors: List[Dict[str, Any]],
    ) -> str:
        """
        Generate coaching feedback from action segments and posture errors.

        This implements Module 3 (Feedback Generation) from the spec.

        Args:
            action_segments: List of detected action segments from sliding window.
            posture_errors: List of posture error dicts from heuristics engine.

        Returns:
            Coaching feedback text.
        """
        # Create raw sequence of actions
        action_list = [seg["action"] for seg in action_segments]
        
        # Count action occurrences for the analytical fallback
        from collections import Counter
        action_counts = dict(Counter(action_list))

        prompt = PromptTemplates.get_posture_coaching_prompt(
            action_list=action_list,
            posture_errors=posture_errors,
        )

        feedback = self._try_generate(
            prompt=prompt,
            max_new_tokens=150,
            temperature=0.7,
        )
        if feedback is None:
            feedback = self._generate_analytical_posture_feedback(
                action_counts, posture_errors
            )

        return feedback

    @staticmethod
    def _generate_analytical_posture_feedback(
        action_counts: Dict[str, int],
        posture_errors: List[Dict[str, Any]],
    ) -> str:
        """Analytical fallback when LLM is unavailable."""
        parts = []

        # Action summary
        if action_counts:
            action_strs = [
                f"{count}× {action}"
                for action, count in sorted(
                    action_counts.items(), key=lambda x: x[1], reverse=True
                )
            ]
            parts.append(f"Actions detected: {', '.join(action_strs)}.")

        # Error summary
        if posture_errors:
            high = [e for e in posture_errors if e.get("severity") == "high"]
            medium = [e for e in posture_errors if e.get("severity") == "medium"]
            if high:
                parts.append(
                    f"{len(high)} high-severity issue(s): "
                    + "; ".join(e["error"] for e in high)
                    + "."
                )
            if medium:
                parts.append(
                    f"{len(medium)} medium-severity issue(s): "
                    + "; ".join(e["error"] for e in medium)
                    + "."
                )
        else:
            parts.append("No posture errors detected — good form!")

        return " ".join(parts)

