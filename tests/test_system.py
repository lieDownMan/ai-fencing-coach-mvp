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

    def test_pose_estimator_mock_frame_skeleton(self):
        """Test explicit mock backend returns a valid non-zero skeleton."""
        from src.pose_estimation import PoseEstimator

        estimator = PoseEstimator(backend="mock")
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        skeleton = estimator.extract_frame_skeleton(frame)

        assert skeleton is not None
        assert estimator.validate_skeleton(skeleton)
        assert skeleton["front_ankle"] == skeleton["right_ankle"]
        assert any(coord != 0.0 for point in skeleton.values() for coord in point)

    def test_pose_estimator_unavailable_backend_raises(self):
        """Test unavailable real pose backend fails clearly instead of returning zeros."""
        from src.pose_estimation import PoseEstimator

        estimator = PoseEstimator(backend="mock")
        estimator.backend = "unavailable"
        estimator.model = None

        with pytest.raises(RuntimeError, match="Pose estimator is unavailable"):
            estimator.extract_frame_skeleton(np.zeros((10, 10, 3), dtype=np.uint8))

    def test_pose_estimator_validate_rejects_invalid_coordinates(self):
        """Test skeleton validation rejects missing or non-finite coordinates."""
        from src.pose_estimation import PoseEstimator

        estimator = PoseEstimator(backend="mock")
        skeleton = estimator.extract_frame_skeleton(np.zeros((10, 10, 3), dtype=np.uint8))
        skeleton["nose"] = (np.nan, 0.0)

        assert not estimator.validate_skeleton(skeleton)


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

    def test_spatial_normalizer_rejects_invalid_coordinate(self):
        """Test SpatialNormalizer rejects malformed or non-finite coordinates early."""
        from src.preprocessing import SpatialNormalizer

        normalizer = SpatialNormalizer()

        with pytest.raises(ValueError, match="non-finite"):
            normalizer.fit([
                {
                    "nose": (np.nan, 100.0),
                    "front_ankle": (100.0, 200.0),
                }
            ])
    
    def test_temporal_sampler_initialization(self):
        """Test TemporalSampler initialization."""
        from src.preprocessing import TemporalSampler
        
        sampler = TemporalSampler(target_length=28)
        assert sampler.target_length == 28

    def test_temporal_sampler_rejects_invalid_array_shape(self):
        """Test TemporalSampler rejects arrays that are not frame/joint/xy tensors."""
        from src.preprocessing import TemporalSampler

        sampler = TemporalSampler(target_length=28)

        with pytest.raises(ValueError, match="shape"):
            sampler.sample_array(np.zeros((28, 20), dtype=np.float32))

    def test_temporal_sampler_repeats_single_frame_array(self):
        """Test one-frame sequences are safely expanded to the target length."""
        from src.preprocessing import TemporalSampler

        sampler = TemporalSampler(target_length=4)
        skeleton_array = np.array([[[1.0, 2.0], [3.0, 4.0]]])

        sampled = sampler.sample_array(skeleton_array)

        assert sampled.shape == (4, 2, 2)
        assert np.allclose(sampled[0], skeleton_array[0])
        assert np.allclose(sampled[-1], skeleton_array[0])

    def test_temporal_sampler_rejects_inconsistent_skeleton_keys(self):
        """Test list-based interpolation fails clearly on inconsistent joints."""
        from src.preprocessing import TemporalSampler

        sampler = TemporalSampler(target_length=4)
        sequence = [
            {"nose": (0.0, 0.0), "front_ankle": (0.0, 1.0)},
            {"nose": (1.0, 0.0)},
        ]

        with pytest.raises(KeyError, match="inconsistent"):
            sampler.sample(sequence)

    def test_model_joint_array_has_twenty_channels(self):
        """Test that the model feature joint set excludes normalization-only aliases."""
        from src.preprocessing import SpatialNormalizer

        skeleton = {
            joint_name: (float(idx), float(idx + 10))
            for idx, joint_name in enumerate(SpatialNormalizer.MODEL_JOINT_NAMES)
        }
        skeleton["nose"] = (0.0, 0.0)
        skeleton["front_ankle"] = (0.0, 100.0)

        normalizer = SpatialNormalizer()
        normalizer.fit([skeleton])
        array = normalizer.get_normalized_array(
            [skeleton],
            joint_names=SpatialNormalizer.MODEL_JOINT_NAMES
        )

        assert array.shape == (1, 10, 2)
        assert array.shape[1] * array.shape[2] == 20


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

    def test_tcn_block_preserves_time_length(self):
        """Test TCNBlock keeps sequence length stable across dilations."""
        from src.models import TCNBlock

        for dilation in (1, 2, 4):
            tcn = TCNBlock(in_channels=20, out_channels=20, dilation=dilation)
            x = torch.randn(2, 20, 28)
            output = tcn(x)

            assert output.shape == x.shape
    
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

    def test_bifencenet_forward(self):
        """Test BiFenceNet forward pass."""
        from src.models import BiFenceNet

        model = BiFenceNet(input_channels=20, hidden_channels=64)
        x = torch.randn(2, 20, 28)
        logits = model(x)

        assert logits.shape == (2, 6)


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

    def test_pattern_analyzer_rejects_invalid_confidence(self):
        """Test PatternAnalyzer validates confidence values."""
        from src.tracking import PatternAnalyzer

        analyzer = PatternAnalyzer()

        with pytest.raises(ValueError, match="confidence"):
            analyzer.add_classification(0, 1.5)
    
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

    def test_pattern_analyzer_transitions_are_plain_dicts(self):
        """Test transition summaries are JSON-friendly plain dictionaries."""
        from src.tracking import PatternAnalyzer

        analyzer = PatternAnalyzer()
        analyzer.add_classification(0, 0.9)
        analyzer.add_classification(4, 0.9)
        analyzer.add_classification(0, 0.9)

        transitions = analyzer.get_action_transitions()

        assert transitions == {"R": {"SF": 1}, "SF": {"R": 1}}
        assert type(transitions["R"]) is dict
    
    def test_profile_manager_initialization(self):
        """Test ProfileManager initialization."""
        from src.tracking import ProfileManager
        from pathlib import Path
        import tempfile
        
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ProfileManager(profiles_dir=tmpdir)
            assert manager is not None
            assert Path(tmpdir).exists()

    def test_profile_manager_completed_result_is_not_loss(self):
        """Test non-competitive completed bouts do not count as losses."""
        from src.tracking import ProfileManager
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ProfileManager(profiles_dir=tmpdir)
            profile = manager.save_bout(
                "athlete_001",
                {"offensive_ratio": 0.5, "defensive_ratio": 0.25, "js_sf_ratio": 1.0},
                result="completed"
            )

        stats = profile["overall_stats"]
        assert stats["total_bouts"] == 1
        assert stats["wins"] == 0
        assert stats["losses"] == 0
        assert stats["completed_bouts"] == 1

    def test_profile_manager_counts_explicit_wins_and_losses(self):
        """Test ProfileManager only counts explicit wins/losses as such."""
        from src.tracking import ProfileManager
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ProfileManager(profiles_dir=tmpdir)
            manager.save_bout("athlete_001", {"offensive_ratio": 1.0}, result="win")
            profile = manager.save_bout("athlete_001", {"offensive_ratio": 0.0}, result="loss")

        stats = profile["overall_stats"]
        assert stats["wins"] == 1
        assert stats["losses"] == 1
        assert stats["total_bouts"] == 2
        assert stats["average_offensive_ratio"] == 0.5


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
        assert pipeline.model_input_channels == 20
        assert hasattr(pipeline, 'process_video')

    def test_system_pipeline_rejects_channel_mismatch(self):
        """Test that Phase 3 fails fast when Phase 2 emits the wrong channel count."""
        from src.app_interface import SystemPipeline

        pipeline = SystemPipeline(device="cpu", use_bifencenet=False)
        skeleton_array = np.zeros((28, 11, 2), dtype=np.float32)

        with pytest.raises(ValueError, match="channel mismatch"):
            pipeline._run_inference(skeleton_array)

    def test_system_pipeline_inference_batches_windows(self):
        """Test Phase 3 batch inference returns one classification per sliding window."""
        from src.app_interface import SystemPipeline
        import tempfile

        pipeline = SystemPipeline(
            device="cpu",
            use_bifencenet=False,
            profiles_dir=tempfile.mkdtemp(),
            pose_backend="mock"
        )
        skeleton_array = np.zeros((56, 10, 2), dtype=np.float32)

        classifications = pipeline._run_inference(
            skeleton_array,
            batch_process=True,
            batch_size=2
        )

        assert len(classifications) == 3
        assert all(0 <= class_idx < 6 for class_idx, _ in classifications)
        assert all(0.0 <= confidence <= 1.0 for _, confidence in classifications)

    def test_system_pipeline_inference_rejects_non_finite_values(self):
        """Test Phase 3 rejects NaN/inf model inputs."""
        from src.app_interface import SystemPipeline
        import tempfile

        pipeline = SystemPipeline(
            device="cpu",
            use_bifencenet=False,
            profiles_dir=tempfile.mkdtemp(),
            pose_backend="mock"
        )
        skeleton_array = np.zeros((28, 10, 2), dtype=np.float32)
        skeleton_array[0, 0, 0] = np.nan

        with pytest.raises(ValueError, match="non-finite"):
            pipeline._run_inference(skeleton_array)

    def test_system_pipeline_loads_state_dict_checkpoint(self):
        """Test Phase 3 accepts common checkpoint dictionaries."""
        from src.app_interface import SystemPipeline
        from src.models import FenceNet
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "model.pt"
            model = FenceNet(input_channels=20, hidden_channels=64)
            torch.save({"state_dict": model.state_dict()}, checkpoint_path)

            pipeline = SystemPipeline(
                device="cpu",
                use_bifencenet=False,
                model_checkpoint=str(checkpoint_path),
                profiles_dir=str(Path(tmpdir) / "profiles"),
                pose_backend="mock"
            )

        assert pipeline.model.training is False

    def test_system_pipeline_preserves_long_sequences_for_sliding_windows(self):
        """Test Phase 2 keeps long videos long enough for sliding-window inference."""
        from src.app_interface import SystemPipeline
        from src.preprocessing import SpatialNormalizer
        import tempfile

        pipeline = SystemPipeline(
            device="cpu",
            use_bifencenet=False,
            profiles_dir=tempfile.mkdtemp(),
            pose_backend="mock"
        )

        skeleton_array = np.zeros((56, 10, 2), dtype=np.float32)
        calls = []

        def fake_get_normalized_array(*args, **kwargs):
            return skeleton_array

        def fake_run_inference(array, batch_process=True):
            calls.append(array.shape)
            return [(0, 1.0)] * 3

        pipeline.pose_estimator.extract_video_skeleton = lambda _: [
            {
                **{joint: (float(idx), float(idx + 1)) for idx, joint in enumerate(SpatialNormalizer.MODEL_JOINT_NAMES)},
                "front_ankle": (0.0, 100.0),
            }
            for _ in range(56)
        ]
        pipeline.spatial_normalizer.get_normalized_array = fake_get_normalized_array
        pipeline._run_inference = fake_run_inference

        results = pipeline.process_video("synthetic.mp4", "athlete_001")

        assert calls == [(56, 10, 2)]
        assert len(results["classifications"]) == 3

    def test_system_pipeline_resets_pattern_history_per_video(self):
        """Test Phase 4 statistics are per-video, not accumulated across videos."""
        from src.app_interface import SystemPipeline
        from src.preprocessing import SpatialNormalizer
        import tempfile

        pipeline = SystemPipeline(
            device="cpu",
            use_bifencenet=False,
            profiles_dir=tempfile.mkdtemp(),
            pose_backend="mock"
        )
        pipeline.pose_estimator.extract_video_skeleton = lambda _: [
            {
                **{joint: (float(idx), float(idx + 1)) for idx, joint in enumerate(SpatialNormalizer.MODEL_JOINT_NAMES)},
                "front_ankle": (0.0, 100.0),
            }
            for _ in range(28)
        ]
        pipeline._run_inference = lambda *args, **kwargs: [(0, 0.9)]

        first = pipeline.process_video("first.mp4", "athlete_001")
        second = pipeline.process_video("second.mp4", "athlete_001")

        assert first["statistics"]["total_actions"] == 1
        assert second["statistics"]["total_actions"] == 1
    
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
            normalizer.get_normalized_array(normalized, already_normalized=True)
        )
        
        assert resampled.shape == (28, len(normalized[0]), 2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
