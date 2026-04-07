"""AI Fencing Coach & Referee System - Main package"""
from src.pose_estimation import PoseEstimator
from src.preprocessing import SpatialNormalizer, TemporalSampler
from src.models import FenceNet, BiFenceNet, TCNBlock
from src.tracking import FencerTracker, PatternAnalyzer, ProfileManager
from src.llm_agent import CoachEngine, PromptTemplates, ModelLoader
from src.app_interface import SystemPipeline, FencingCoachUI

__all__ = [
    "PoseEstimator",
    "SpatialNormalizer",
    "TemporalSampler",
    "FenceNet",
    "BiFenceNet",
    "TCNBlock",
    "FencerTracker",
    "PatternAnalyzer",
    "ProfileManager",
    "CoachEngine",
    "PromptTemplates",
    "ModelLoader",
    "SystemPipeline",
    "FencingCoachUI",
]

__version__ = "0.1.0"
