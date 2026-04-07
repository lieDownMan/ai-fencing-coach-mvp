"""
Tests for the AI Fencing Coach System
"""

import json
import pytest
import numpy as np
import torch
from pathlib import Path


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

    def test_prompt_templates_conclusive_uses_average_confidence(self):
        """Test conclusive prompt uses PatternAnalyzer's confidence field."""
        from src.llm_agent import PromptTemplates

        prompt = PromptTemplates.get_conclusive_feedback_prompt(
            bout_stats={
                "defensive_ratio": 0.25,
                "offensive_ratio": 0.75,
                "js_sf_ratio": 1.0,
                "average_confidence": 0.75,
            },
            historical_progression={},
            bout_result="completed"
        )

        assert "Average Action Confidence: 75.0%" in prompt

    def test_model_loader_reports_unimplemented_backend(self):
        """Test ModelLoader does not pretend the placeholder LLM is loaded."""
        from src.llm_agent import ModelLoader

        loader = ModelLoader()

        assert loader.load_model() is False
        assert not loader.is_loaded()
        assert loader.get_model_info()["load_attempted"] is True
        with pytest.raises(RuntimeError, match="Model not loaded"):
            loader.generate("Give fencing feedback")
    
    def test_coach_engine_initialization(self):
        """Test CoachEngine initialization."""
        from src.llm_agent import CoachEngine
        
        engine = CoachEngine()
        assert engine is not None
        assert hasattr(engine, 'generate_immediate_feedback')
        assert hasattr(engine, 'generate_break_strategy')

    def test_coach_engine_immediate_feedback_uses_passed_stats(self, tmp_path):
        """Test immediate feedback can use active pipeline stats as fallback input."""
        from src.llm_agent import CoachEngine

        engine = CoachEngine(profiles_dir=str(tmp_path))
        stats = {
            "action_frequencies": {"R": 0.75, "SB": 0.25},
            "offensive_ratio": 0.75,
            "defensive_ratio": 0.25,
            "repetitive_patterns": [],
        }

        feedback = engine.generate_immediate_feedback(
            fencer_id="athlete_001",
            current_score={"player": 3, "opponent": 2},
            stats=stats,
            recent_actions=["R", "R", "SB", "R"]
        )

        assert "R: 75%" in feedback
        assert "Maintain aggressive momentum" in feedback


class TestAppInterface:
    """Test suite for application interface."""

    def test_load_app_config_reads_yaml(self, tmp_path):
        """Test app config loader reads YAML mappings."""
        from app import load_app_config

        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            "model:\n  type: bifencenet\nathlete:\n  default_id: athlete_cfg\n",
            encoding="utf-8"
        )

        config = load_app_config(config_path)

        assert config["model"]["type"] == "bifencenet"
        assert config["athlete"]["default_id"] == "athlete_cfg"

    def test_write_json_report_contains_required_video_summary(self, tmp_path):
        """Test JSON reports include frame count, windows, stats, and feedback."""
        from app import write_json_report

        results = {
            "ok": True,
            "video_path": "clip.mp4",
            "fencer_id": "athlete_001",
            "frames_processed": 42,
            "window_size": 28,
            "window_stride": 14,
            "classifications": [(0, 0.9), (5, 0.25)],
            "statistics": {
                "total_actions": 2,
                "action_frequencies": {"R": 0.5, "SB": 0.5},
                "offensive_ratio": 0.5,
                "defensive_ratio": 0.5,
                "js_sf_ratio": 0.0,
                "repetitive_patterns": [],
                "average_confidence": 0.575,
            },
            "feedback": "Practice recovering after lunges.",
        }

        report_path = write_json_report(
            results,
            output_path=tmp_path / "report.json",
            runtime_metadata={"pose_backend": "mock", "model_weights": "random"}
        )
        report = json.loads(report_path.read_text(encoding="utf-8"))

        assert report["frames_processed"] == 42
        assert report["classification_window_count"] == 2
        assert report["classification_windows"][0]["action"] == "R"
        assert report["classification_windows"][1]["window_start_frame"] == 14
        assert report["statistics"]["action_frequencies"]["SB"] == 0.5
        assert report["statistics"]["average_confidence"] == 0.575
        assert report["feedback"] == "Practice recovering after lunges."
        assert report["runtime"]["pose_backend"] == "mock"

    def test_main_writes_json_report_from_cli_flag(
        self,
        tmp_path,
        monkeypatch,
        capsys
    ):
        """Test CLI --report writes a processed-video report."""
        import app as app_module

        video_path = tmp_path / "clip.mp4"
        video_path.write_bytes(b"not a real video; app is mocked")
        report_path = tmp_path / "reports" / "clip_report.json"

        class FakeApplication:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def process_video(self, **kwargs):
                return {
                    "ok": True,
                    "video_path": kwargs["video_path"],
                    "fencer_id": kwargs["fencer_id"],
                    "frames_processed": 28,
                    "window_size": 28,
                    "window_stride": 14,
                    "classifications": [(4, 0.8)],
                    "statistics": {
                        "total_actions": 1,
                        "action_frequencies": {"SF": 1.0},
                        "offensive_ratio": 0.0,
                        "defensive_ratio": 0.0,
                        "js_sf_ratio": 0.0,
                        "repetitive_patterns": [],
                        "average_confidence": 0.8,
                    },
                    "feedback": "mock feedback",
                }

            def get_runtime_metadata(self):
                return {"pose_backend": "mock", "model_weights": "random"}

        monkeypatch.setattr(app_module, "FencingCoachApplication", FakeApplication)

        exit_code = app_module.main([
            "--video",
            str(video_path),
            "--fencer-id",
            "cli_fencer",
            "--report",
            str(report_path),
        ])
        output = capsys.readouterr().out
        report = json.loads(report_path.read_text(encoding="utf-8"))

        assert exit_code == 0
        assert "Report written:" in output
        assert report["fencer_id"] == "cli_fencer"
        assert report["classification_windows"][0]["action"] == "SF"
        assert report["runtime"]["model_weights"] == "random"

    def test_main_missing_video_returns_nonzero_without_initializing_app(
        self,
        tmp_path,
        monkeypatch,
        capsys
    ):
        """Test CLI fails fast on a missing video before app initialization."""
        import app as app_module

        class ExplodingApplication:
            def __init__(self, *args, **kwargs):
                raise AssertionError("Application should not be initialized")

        monkeypatch.setattr(
            app_module,
            "FencingCoachApplication",
            ExplodingApplication
        )

        exit_code = app_module.main(["--video", str(tmp_path / "missing.mp4")])
        output = capsys.readouterr().out

        assert exit_code == 1
        assert "Video file not found" in output

    def test_main_uses_config_defaults_for_app(
        self,
        tmp_path,
        monkeypatch,
        capsys
    ):
        """Test CLI wires config defaults into the application constructor."""
        import app as app_module

        video_path = tmp_path / "clip.mp4"
        video_path.write_bytes(b"not a real video; app is mocked")
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            "\n".join([
                "model:",
                "  type: bifencenet",
                "data:",
                "  profiles_dir: profiles_from_config",
                "llm:",
                "  model_name: mistral",
                "  device: cpu",
                "pose:",
                "  backend: mock",
                "athlete:",
                "  default_id: athlete_cfg",
                "ui:",
                "  window_width: 320",
                "  window_height: 240",
            ]),
            encoding="utf-8"
        )
        captured = {}

        class FakeApplication:
            def __init__(self, **kwargs):
                captured["app_kwargs"] = kwargs

            def process_video(self, **kwargs):
                captured["process_kwargs"] = kwargs
                return {
                    "ok": True,
                    "video_path": kwargs["video_path"],
                    "frames_processed": 0,
                    "feedback": "mock feedback",
                }

        monkeypatch.setattr(app_module, "FencingCoachApplication", FakeApplication)

        exit_code = app_module.main([
            "--config",
            str(config_path),
            "--video",
            str(video_path)
        ])
        output = capsys.readouterr().out

        assert exit_code == 0
        assert captured["app_kwargs"]["use_bifencenet"] is True
        assert captured["app_kwargs"]["profiles_dir"] == "profiles_from_config"
        assert captured["app_kwargs"]["llm_model_name"] == "mistral"
        assert captured["app_kwargs"]["device"] == "cpu"
        assert captured["app_kwargs"]["pose_backend"] == "mock"
        assert captured["app_kwargs"]["ui_width"] == 320
        assert captured["app_kwargs"]["ui_height"] == 240
        assert captured["process_kwargs"]["fencer_id"] == "athlete_cfg"
        assert "mock feedback" in output

    def test_main_cli_flags_override_config_defaults(
        self,
        tmp_path,
        monkeypatch
    ):
        """Test explicit CLI flags take precedence over config defaults."""
        import app as app_module

        video_path = tmp_path / "clip.mp4"
        video_path.write_bytes(b"not a real video; app is mocked")
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            "\n".join([
                "model:",
                "  type: fencenet",
                "data:",
                "  profiles_dir: config_profiles",
                "llm:",
                "  model_name: llava-next",
                "  device: auto",
                "pose:",
                "  backend: auto",
                "athlete:",
                "  default_id: config_fencer",
            ]),
            encoding="utf-8"
        )
        captured = {}

        class FakeApplication:
            def __init__(self, **kwargs):
                captured["app_kwargs"] = kwargs

            def process_video(self, **kwargs):
                captured["process_kwargs"] = kwargs
                return {
                    "ok": True,
                    "video_path": kwargs["video_path"],
                    "frames_processed": 0,
                    "feedback": "",
                }

        monkeypatch.setattr(app_module, "FencingCoachApplication", FakeApplication)

        exit_code = app_module.main([
            "--config",
            str(config_path),
            "--video",
            str(video_path),
            "--fencer-id",
            "cli_fencer",
            "--profiles-dir",
            "cli_profiles",
            "--pose-backend",
            "mock",
            "--llm-model",
            "mistral",
            "--device",
            "cpu",
            "--model-type",
            "bifencenet",
        ])

        assert exit_code == 0
        assert captured["app_kwargs"]["use_bifencenet"] is True
        assert captured["app_kwargs"]["profiles_dir"] == "cli_profiles"
        assert captured["app_kwargs"]["pose_backend"] == "mock"
        assert captured["app_kwargs"]["llm_model_name"] == "mistral"
        assert captured["app_kwargs"]["device"] == "cpu"
        assert captured["process_kwargs"]["fencer_id"] == "cli_fencer"
    
    def test_system_pipeline_initialization(self):
        """Test SystemPipeline initialization."""
        from src.app_interface import SystemPipeline
        
        pipeline = SystemPipeline(device="cpu", use_bifencenet=False)
        assert pipeline is not None
        assert pipeline.device == "cpu"
        assert pipeline.model_input_channels == 20
        assert hasattr(pipeline, 'process_video')

    def test_application_missing_video_returns_error_payload(self, tmp_path):
        """Test application process_video returns a stable missing-file error."""
        from app import FencingCoachApplication

        app = FencingCoachApplication(
            device="cpu",
            profiles_dir=str(tmp_path),
            pose_backend="mock"
        )

        result = app.process_video(
            video_path=str(tmp_path / "missing.mp4"),
            fencer_id="athlete_001"
        )

        assert result["ok"] is False
        assert "Video file not found" in result["error"]

    def test_system_pipeline_immediate_feedback_uses_pipeline_stats(self, tmp_path):
        """Test Phase 5 feedback is based on the pipeline's real analyzer state."""
        from src.app_interface import SystemPipeline

        pipeline = SystemPipeline(
            device="cpu",
            use_bifencenet=False,
            profiles_dir=str(tmp_path),
            pose_backend="mock"
        )
        pipeline.set_fencer("athlete_001")
        for class_idx in (0, 0, 0, 5):
            pipeline.pattern_analyzer.add_classification(class_idx, 0.9)

        feedback = pipeline.get_immediate_feedback()

        assert "R: 75%" in feedback
        assert "Maintain aggressive momentum" in feedback

    def test_system_pipeline_break_strategy_uses_pipeline_stats(self, tmp_path):
        """Test break strategy uses active analyzer stats instead of empty coach state."""
        from src.app_interface import SystemPipeline

        pipeline = SystemPipeline(
            device="cpu",
            use_bifencenet=False,
            profiles_dir=str(tmp_path),
            pose_backend="mock"
        )
        pipeline.set_fencer("athlete_001")
        for class_idx in (4, 4, 4):
            pipeline.pattern_analyzer.add_classification(class_idx, 0.9)

        strategy = pipeline.get_break_strategy()

        assert "collect a few exchanges first" not in strategy
        assert "Increase variety" in strategy

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

    def test_fencing_coach_ui_headless_display_skips_opencv_window(
        self,
        monkeypatch
    ):
        """Test display can render without opening a window in CI."""
        from src.app_interface import main_ui

        def fail_if_called(*args, **kwargs):
            raise AssertionError("OpenCV window function should not be called")

        monkeypatch.setattr(main_ui.cv2, "imshow", fail_if_called)
        monkeypatch.setattr(main_ui.cv2, "waitKey", fail_if_called)

        ui = main_ui.FencingCoachUI(
            width=320,
            height=240,
            display_enabled=False
        )
        canvas = ui.display()
        ui.close()

        assert canvas.shape == (240, 320, 3)


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

    def test_sample_video_mock_pipeline_smoke(self, tmp_path):
        """Run the ignored local sample video through the mock full pipeline when present."""
        import cv2
        from app import FencingCoachApplication

        video_path = Path("video/fencing_match.mp4")
        if not video_path.exists():
            pytest.skip("Local sample video is not present")

        cap = cv2.VideoCapture(str(video_path))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if frame_count <= 0:
            pytest.skip("Local sample video cannot be read by OpenCV")
        if frame_count < 28:
            pytest.skip("Local sample video is shorter than one model window")

        app = FencingCoachApplication(
            device="cpu",
            profiles_dir=str(tmp_path),
            pose_backend="mock"
        )
        results = app.process_video(
            video_path=str(video_path),
            fencer_id="sample_video_smoke"
        )

        expected_windows = ((frame_count - 28) // 14) + 1
        assert results["ok"] is True
        assert results["frames_processed"] == frame_count
        assert len(results["classifications"]) == expected_windows
        assert results["statistics"]["total_actions"] == expected_windows
        assert results["feedback"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
