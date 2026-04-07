"""
Prompt Templates - System prompts for different coaching feedback types.
"""

from typing import Dict, List, Any
from enum import Enum


class FeedbackType(Enum):
    """Types of coaching feedback."""
    IMMEDIATE = "immediate"
    BREAK_STRATEGY = "break_strategy"
    CONCLUSIVE = "conclusive"


class PromptTemplates:
    """
    Generates system prompts for different types of coaching feedback.
    """
    
    IMMEDIATE_SYSTEM_PROMPT = """You are an expert fencing coach providing immediate tactical feedback during a bout.
Your feedback must be:
- EXTREMELY CONCISE (1-2 sentences maximum)
- Actionable and specific to current performance
- Focused on immediate technical adjustments
- Encouraging but honest

Respond with only the feedback, no explanations."""

    BREAK_STRATEGY_SYSTEM_PROMPT = """You are an expert fencing coach providing strategic advice during the 1-minute break between bouts.
Your feedback should:
- Analyze opponent's tactical patterns and tendencies
- Suggest defensive or offensive adjustments
- Be practical and implementable in next bout
- Consider current score and bout situation
- Be concise but comprehensive (2-3 sentences)

Consider the opponent profile and suggest specific tactical adjustments."""

    CONCLUSIVE_SYSTEM_PROMPT = """You are an expert fencing coach providing post-bout analysis and progression feedback.
Your analysis should:
- Compare current performance with historical data
- Highlight improvements and areas for development
- Provide specific training recommendations
- Celebrate accomplishments
- Be constructive and motivating
- Include tactical and technical insights

Format your response with clear sections for strengths, improvements, and next steps."""

    @staticmethod
    def get_immediate_feedback_prompt(
        action_frequencies: Dict[str, float],
        offensive_ratio: float,
        defensive_ratio: float,
        recent_actions: List[str],
        current_score: Dict[str, int],
        patterns: List = None
    ) -> str:
        """
        Generate immediate feedback prompt.
        
        Args:
            action_frequencies: Dictionary of action frequencies
            offensive_ratio: Ratio of offensive moves
            defensive_ratio: Ratio of defensive moves
            recent_actions: Last N actions performed
            current_score: Current bout score
            patterns: Detected patterns
            
        Returns:
            Formatted prompt for LLM
        """
        top_actions = ", ".join(
            f"{action}: {frequency:.0%}"
            for action, frequency in sorted(
                action_frequencies.items(),
                key=lambda x: x[1],
                reverse=True
            )[:3]
        )

        context = f"""Current Bout Status:
- Score: Your {current_score.get('player', 0)} - Opponent {current_score.get('opponent', 0)}
- Recent Actions: {', '.join(recent_actions[-5:])}
- Offensive Ratio: {offensive_ratio:.1%}
- Defensive Ratio: {defensive_ratio:.1%}
- Top Actions: {top_actions}
"""
        
        if patterns:
            context += f"- Detected Patterns: {patterns}\n"
        
        context += "\nGive immediate tactical feedback for the next exchange."
        
        return context

    @staticmethod
    def get_break_strategy_prompt(
        fencer_stats: Dict[str, Any],
        opponent_patterns: Dict[str, Any],
        js_sf_ratio: float,
        current_score: Dict[str, int]
    ) -> str:
        """
        Generate break strategy prompt.
        
        Args:
            fencer_stats: Current fencer statistics
            opponent_patterns: Detected opponent patterns
            js_sf_ratio: JS/SF ratio
            current_score: Current bout score
            
        Returns:
            Formatted prompt for LLM
        """
        context = f"""Break-Time Strategic Analysis:

Your Performance:
- Offensive Actions: {fencer_stats.get('offensive_ratio', 0):.1%}
- Jumping Sliding / Step Forward Ratio: {js_sf_ratio:.2f}
- Most Frequent Action: {fencer_stats.get('most_frequent_action', 'Unknown')}

Opponent Tendencies:
- Primarily Defensive: {opponent_patterns.get('defensive_ratio', 0):.1%}
- Common Patterns: {opponent_patterns.get('patterns', [])}

Score Status: Your {current_score.get('player', 0)} - Opponent {current_score.get('opponent', 0)}

Provide tactical adjustments for the next minute of bout."""
        
        return context

    @staticmethod
    def get_conclusive_feedback_prompt(
        bout_stats: Dict[str, Any],
        historical_progression: Dict[str, Any],
        bout_result: str,
        opponent_info: Dict[str, Any] = None
    ) -> str:
        """
        Generate post-bout conclusive feedback prompt.
        
        Args:
            bout_stats: Statistics from this bout
            historical_progression: Historical performance data
            bout_result: Bout result (win/loss/draw)
            opponent_info: Information about opponent
            
        Returns:
            Formatted prompt for LLM
        """
        context = f"""Post-Bout Performance Analysis:

Bout Result: {bout_result.upper()}

This Bout Metrics:
- Defensive Ratio: {bout_stats.get('defensive_ratio', 0):.1%}
- Offensive Ratio: {bout_stats.get('offensive_ratio', 0):.1%}
- JS/SF Ratio: {bout_stats.get('js_sf_ratio', 0):.2f}
- Average Action Confidence: {bout_stats.get('avg_confidence', 0):.1%}

Historical Comparison:
- Previous Average Offensive Ratio: {historical_progression.get('avg_offensive', 0):.1%}
- Recent Bout Trends: {historical_progression.get('trends', [])}
- Overall Win Rate: {historical_progression.get('win_rate', 0):.1%}

"""
        
        if opponent_info:
            context += f"Opponent Level: {opponent_info.get('level', 'Unknown')}\n"
        
        context += "\nProvide comprehensive post-bout analysis with areas of improvement and specific training recommendations."
        
        return context

    @staticmethod
    def format_action_summary(
        action_frequencies: Dict[str, float],
        max_items: int = 5
    ) -> str:
        """
        Format action frequency summary.
        
        Args:
            action_frequencies: Dictionary of action frequencies
            max_items: Maximum number of items to show
            
        Returns:
            Formatted string
        """
        sorted_actions = sorted(
            action_frequencies.items(),
            key=lambda x: x[1],
            reverse=True
        )[:max_items]
        
        summary = "Action Summary:\n"
        for action, freq in sorted_actions:
            summary += f"- {action}: {freq:.1%}\n"
        
        return summary
