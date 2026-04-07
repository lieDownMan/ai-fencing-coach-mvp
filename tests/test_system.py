"""
Tests for the AI Fencing Coach System
"""

import pytest
import numpy as np
import torch


class TestPoseEstimation:
    """Test suite for pose estimation module."""
    
    def test_pose_estimator_initialization(self):
        """Test PoseEstimator initialization."""
        from src.pose_estimation import PoseEstimator
        
        estimator = PoseEstimator()
        assert estimator is not None
        assert hasattr(estimator, 'extract_frame_skeleton')
        assert hasattr(estimator, 'extract_video_skeleton')


class TestPreprocessing:
    """Test suite for preprocessing module."""
    
    def test_spatial_normalizer_initialization(self):
        """Test SpatialNormalizer initialization."""
        from src.preprocessing import SpatialNormalizer
        
        normalizer = SpatialNormalizer()
        assert normalizer is not None
        assert normalizer.reference_nose is None
        assert normalizer.scale_factor is None
    
    def test_spatial_normalizer_fit(self):
        """Test SpatialNormalizer fitting."""
        from src.preprocessing import SpatialNormalizer
        
        normalizer = SpatialNormalizer()
        skeleton_seq = [
            {
                "nose": (100.0, 100.0),
                "front_ankle": (100.0, 200.0),
                "left_hip": (90.0, 150.0),
                "right_hip": (110.0, 150.0),
            },
            {
                "nose": (105.0, 100.0),
                "front_ankle": (105.0, 200.0),
                "left_hip": (95.0, 150.0),
                "right_hip": (115.0, 150.0),
            }
        ]
        
        normalizer.fit(skeleton_seq)
        assert normalizer.scale_factor == 100.0
        assert np.allclose(normalizer.reference_nose, [100.0, 100.0])
    
    def test_temporal_sampler_initialization(self):
        """Test TemporalSampler initialization."""
        from src.preprocessing import TemporalSampler
        
        sampler = TemporalSampler(target_length=28)
        assert sampler.target_length == 28


class TestModels:
    """Test suite for model architectures."""
    
    def test_tcn_block_initialization(self):
        """Test TCNBlock initialization."""
        from src.models import TCNBlock
        
        tcn = TCNBlock(
            in_channels=20,
            out_channels=64,
            kernel_size=3,
            dilation=1
        )
        assert tcn is not None
        assert isinstance(tcn, torch.nn.Module)
    
    def test_tcn_block_forward(self):
        """Test TCNBlock forward pass."""
        from src.models import TCNBlock
        
        tcn = TCNBlock(in_channels=20, out_channels=64)
        x = torch.randn(2, 20, 28)  # batch, channels, time
        output = tcn(x)
        
        assert output.shape[0] == 2  # batch size
        assert output.shape[1] == 64  # output channels
    
    def test_fencenet_initialization(self):
        """Test FenceNet initialization."""
        from src.models import FenceNet
        
        model = FenceNet(input_channels=20, hidden_channels=64)
        assert model is not None
        assert isinstance(model, torch.nn.Module)
        assert model.NUM_CLASSES == 6
    
    def test_fencenet_forward(self):
        """Test FenceNet forward pass."""
        from src.models import FenceNet
        
        model = FenceNet(input_channels=20, hidden_channels=64)
        x = torch.randn(2, 20, 28)  # batch, channels, time
        logits = model(x)
        
        assert logits.shape == (2, 6)  # batch, num_classes
    
    def test_bifencenet_initialization(self):
        """Test BiFenceNet initialization."""
        from src.models import BiFenceNet
        
        model = BiFenceNet(input_channels=20, hidden_channels=64)
        assert model is not None
        assert isinstance(model, torch.nn.Module)


class TestTracking:
    """Test suite for tracking module."""
    
    def test_pattern_analyzer_initialization(self):
        """Test PatternAnalyzer initialization."""
        from src.tracking import PatternAnalyzer
        
        analyzer = PatternAnalyzer()
        assert analyzer is not None
        assert len(analyzer.action_history) == 0
    
    def test_pattern_analyzer_add_classification(self):
        """Test adding classifications to PatternAnalyzer."""
        from src.tracking import PatternAnalyzer
        
        analyzer = PatternAnalyzer()
        analyzer.add_classification(0, 0.95)  # R action
        analyzer.add_classification(3, 0.87)  # JS action
        
        assert len(analyzer.action_history) == 2
        assert analyzer.action_history[0] == "R"
        assert analyzer.action_history[1] == "JS"
    
    def test_pattern_analyzer_statistics(self):
        """Test pattern analyzer statistics."""
        from src.tracking import PatternAnalyzer
        
        analyzer = PatternAnalyzer()
        for _ in range(5):
            analyzer.add_classification(0, 0.9)  # 5x R
        for _ in range(3):
            analyzer.add_classification(5, 0.9)  # 3x SB
        
        stats = analyzer.get_statistics_summary()
        freqs = stats['action_frequencies']
        
        assert freqs['R'] == 0.625  # 5/8
        assert freqs['SB'] == 0.375  # 3/8
    
    def test_profile_manager_initialization(self):
        """Test ProfileManager initialization."""
        from src.tracking import ProfileManager
        from pathlib import Path
        import tempfile
        
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ProfileManager(profiles_dir=tmpdir)
            assert manager is not None
            assert Path(tmpdir).exists()


class TestLLMAgent:
    """Test suite for LLM agent module."""
    
    def test_prompt_templates_immediate_feedback(self):
        """Test immediate feedback prompt generation."""
        from src.llm_agent import PromptTemplates
        
        prompt = PromptTemplates.get_immediate_feedback_prompt(
            action_frequencies={'R': 0.3, 'SB': 0.2},
            offensive_ratio=0.6,
            defensive_ratio=0.2,
            recent_actions=['R', 'SF', 'JS'],
            current_score={'player': 2, 'opponent': 1}
        )
        
        assert isinstance(prompt, str)
        assert len(prompt) > 0
        assert 'R: 30%' in prompt or 'R: 0.3' in prompt
    
    def test_coach_engine_initialization(self):
        """Test CoachEngine initialization."""
        from src.llm_agent import CoachEngine
        
        engine = CoachEngine()
        assert engine is not None
        assert hasattr(engine, 'generate_immediate_feedback')
        assert hasattr(engine, 'generate_break_strategy')


class TestAppInterface:
    """Test suite for application interface."""
    
    def test_system_pipeline_initialization(self):
        """Test SystemPipeline initialization."""
        from src.app_interface import SystemPipeline
        
        pipeline = SystemPipeline(device="cpu", use_bifencenet=False)
        assert pipeline is not None
        assert pipeline.device == "cpu"
        assert hasattr(pipeline, 'process_video')
    
    def test_fencing_coach_ui_initialization(self):
        """Test FencingCoachUI initialization."""
        from src.app_interface import FencingCoachUI
        
        ui = FencingCoachUI()
        assert ui is not None
        assert ui.width == 1600
        assert ui.height == 900


# Integration tests
class TestIntegration:
    """Integration tests for the complete system."""
    
    def test_preprocessing_pipeline(self):
        """Test preprocessing pipeline."""
        from src.preprocessing import SpatialNormalizer, TemporalSampler
        import numpy as np
        
        # Create dummy skeleton data
        skeleton_seq = []
        for i in range(50):
            skeleton = {
                "nose": (100 + i*0.1, 100.0),
                "front_ankle": (100 + i*0.1, 200.0),
                "left_hip": (90 + i*0.1, 150.0),
                "right_hip": (110 + i*0.1, 150.0),
                "left_knee": (85 + i*0.1, 175.0),
                "right_knee": (115 + i*0.1, 175.0),
            }
            skeleton_seq.append(skeleton)
        
        # Test normalizer
        normalizer = SpatialNormalizer()
        normalizer.fit(skeleton_seq)
        normalized = normalizer.normalize_sequence(skeleton_seq)
        
        assert len(normalized) == 50
        
        # Test sampler
        sampler = TemporalSampler(target_length=28)
        resampled = sampler.sample_array(
            normalizer.get_normalized_array(normalized)
        )
        
        assert resampled.shape == (28, len(normalized[0]), 2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
