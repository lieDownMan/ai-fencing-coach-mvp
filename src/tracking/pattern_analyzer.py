"""
Pattern Analyzer - Aggregates FenceNet outputs and calculates statistical metrics.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, Counter
import logging

logger = logging.getLogger(__name__)


class PatternAnalyzer:
    """
    Analyzes fencing patterns from real-time FenceNet classifications.
    """
    
    # Action class indices
    ACTION_CLASSES = {
        0: "R",    # Rapid lunge
        1: "IS",   # Incremental speed lunge
        2: "WW",   # With waiting lunge
        3: "JS",   # Jumping sliding lunge
        4: "SF",   # Step forward
        5: "SB",   # Step backward
    }
    
    # Offensive vs Defensive classification
    OFFENSIVE_ACTIONS = {"R", "IS", "WW", "JS"}
    DEFENSIVE_ACTIONS = {"SB"}
    NEUTRAL_ACTIONS = {"SF"}
    
    def __init__(self, window_size: int = 100):
        """
        Initialize Pattern Analyzer.
        
        Args:
            window_size: Size of rolling window for pattern analysis
        """
        self.window_size = window_size
        self.action_history: List[str] = []
        self.timestamps: List[float] = []
        self.confidence_scores: List[float] = []
        
    def add_classification(
        self,
        class_idx: int,
        confidence: float,
        timestamp: Optional[float] = None
    ):
        """
        Add a new classification result.
        
        Args:
            class_idx: Predicted class index (0-5)
            confidence: Confidence score (0-1)
            timestamp: Frame timestamp (optional)
        """
        if class_idx not in self.ACTION_CLASSES:
            raise ValueError(f"Invalid class index: {class_idx}")
        
        action = self.ACTION_CLASSES[class_idx]
        self.action_history.append(action)
        self.confidence_scores.append(confidence)
        
        if timestamp is not None:
            self.timestamps.append(timestamp)
        else:
            self.timestamps.append(len(self.action_history) - 1)
        
        # Keep only recent history
        if len(self.action_history) > self.window_size:
            self.action_history.pop(0)
            self.confidence_scores.pop(0)
            self.timestamps.pop(0)
    
    def get_action_frequencies(self) -> Dict[str, float]:
        """
        Get frequency of each action in recent history.
        
        Returns:
            Dictionary mapping action names to frequencies (0-1)
        """
        if not self.action_history:
            return {action: 0.0 for action in self.ACTION_CLASSES.values()}
        
        total = len(self.action_history)
        counter = Counter(self.action_history)
        
        frequencies = {action: 0.0 for action in self.ACTION_CLASSES.values()}
        for action, count in counter.items():
            frequencies[action] = count / total
        
        return frequencies
    
    def get_offensive_defensive_ratio(self) -> Tuple[float, float]:
        """
        Calculate ratio of offensive to defensive moves.
        
        Returns:
            Tuple of (offensive_ratio, defensive_ratio)
        """
        if not self.action_history:
            return 0.0, 0.0
        
        total = len(self.action_history)
        offensive_count = sum(1 for a in self.action_history if a in self.OFFENSIVE_ACTIONS)
        defensive_count = sum(1 for a in self.action_history if a in self.DEFENSIVE_ACTIONS)
        
        offensive_ratio = offensive_count / total if total > 0 else 0.0
        defensive_ratio = defensive_count / total if total > 0 else 0.0
        
        return offensive_ratio, defensive_ratio
    
    def get_js_sf_ratio(self) -> float:
        """
        Calculate JS (Jumping sliding lunge) vs SF (Step forward) ratio.
        This is a classic fencing footwork metric.
        
        Returns:
            JS/SF ratio
        """
        if not self.action_history:
            return 0.0
        
        js_count = sum(1 for a in self.action_history if a == "JS")
        sf_count = sum(1 for a in self.action_history if a == "SF")
        
        if sf_count == 0:
            return js_count if js_count > 0 else 0.0
        
        return js_count / sf_count
    
    def detect_repetitive_patterns(self, min_repetitions: int = 3) -> List[Tuple[str, int]]:
        """
        Detect repetitive action patterns.
        
        Args:
            min_repetitions: Minimum repetitions to consider as pattern
            
        Returns:
            List of (action_sequence, frequency) tuples
        """
        if not self.action_history:
            return []
        
        patterns = defaultdict(int)
        
        # Look for consecutive repetitions
        i = 0
        while i < len(self.action_history):
            action = self.action_history[i]
            count = 1
            
            j = i + 1
            while j < len(self.action_history) and self.action_history[j] == action:
                count += 1
                j += 1
            
            if count >= min_repetitions:
                pattern_key = f"{action}*{count}"
                patterns[pattern_key] += 1
            
            i = j
        
        return sorted(patterns.items(), key=lambda x: x[1], reverse=True)
    
    def get_average_confidence(self) -> float:
        """Get average confidence score of recent predictions."""
        if not self.confidence_scores:
            return 0.0
        return np.mean(self.confidence_scores)
    
    def get_statistics_summary(self) -> Dict:
        """
        Get comprehensive statistics summary.
        
        Returns:
            Dictionary with various statistics
        """
        freq = self.get_action_frequencies()
        offensive_ratio, defensive_ratio = self.get_offensive_defensive_ratio()
        js_sf_ratio = self.get_js_sf_ratio()
        patterns = self.detect_repetitive_patterns()
        
        summary = {
            "total_actions": len(self.action_history),
            "action_frequencies": freq,
            "offensive_ratio": offensive_ratio,
            "defensive_ratio": defensive_ratio,
            "js_sf_ratio": js_sf_ratio,
            "repetitive_patterns": patterns,
            "average_confidence": self.get_average_confidence(),
        }
        
        return summary
    
    def clear_history(self):
        """Clear action history."""
        self.action_history.clear()
        self.timestamps.clear()
        self.confidence_scores.clear()
    
    def get_action_transitions(self) -> Dict[str, Dict[str, int]]:
        """
        Analyze transitions between actions.
        
        Returns:
            Transition matrix as nested dictionary
        """
        transitions = defaultdict(lambda: defaultdict(int))
        
        for i in range(len(self.action_history) - 1):
            current = self.action_history[i]
            next_action = self.action_history[i + 1]
            transitions[current][next_action] += 1
        
        return dict(transitions)
