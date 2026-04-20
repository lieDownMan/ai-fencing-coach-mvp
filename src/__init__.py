"""AI Fencing Coach & Referee System - Main package"""
from src.pose_estimation import PoseEstimator
from src.preprocessing import SpatialNormalizer, TemporalSampler
from src.models import FenceNet, BiFenceNet, TCNBlock
from src.tracking import FencerTracker, PatternAnalyzer, ProfileManager
from src.llm_agent import CoachEngine, PromptTemplates, ModelLoader
from src.app_interface import SystemPipeline, FencingCoachUI
from src.training import (
    ACTION_CLASSES,
    PreparedDataset,
    TrainingConfig,
    build_dataloaders,
    build_model,
    load_prepared_dataset,
    parse_clip_labels_csv,
    prepare_ffd_dataset,
    prepare_labeled_video_dataset,
    save_prepared_dataset,
    train_model,
)

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
    "ACTION_CLASSES",
    "PreparedDataset",
    "TrainingConfig",
    "build_dataloaders",
    "build_model",
    "load_prepared_dataset",
    "parse_clip_labels_csv",
    "prepare_ffd_dataset",
    "prepare_labeled_video_dataset",
    "save_prepared_dataset",
    "train_model",
]

__version__ = "0.1.0"
