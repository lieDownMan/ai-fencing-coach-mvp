"""Tests for training data preparation and training utilities."""

import csv
from pathlib import Path

import cv2
import numpy as np
import pytest
from scipy.io import savemat


def _write_mock_video(path: Path, frame_count: int = 12):
    """Write a tiny synthetic AVI for mock pose extraction tests."""
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(path), fourcc, 10.0, (320, 240))
    assert writer.isOpened()
    for frame_index in range(frame_count):
        frame = np.full((240, 320, 3), frame_index % 255, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def _synthetic_ffd_body_rows(num_frames: int = 35) -> np.ndarray:
    """Create a synthetic FFD-style n x 161 Kinect body matrix."""
    from src.training.data import KINECT_V1_JOINT_INDEX

    rows = np.zeros((num_frames, 161), dtype=np.float32)
    joint_positions = {
        "head": (0.0, 0.0),
        "shoulder_right": (0.4, 0.3),
        "elbow_right": (0.55, 0.5),
        "wrist_right": (0.7, 0.7),
        "hip_left": (-0.2, 0.9),
        "hip_right": (0.2, 0.9),
        "knee_left": (-0.2, 1.2),
        "knee_right": (0.2, 1.2),
        "ankle_left": (-0.2, 1.6),
        "ankle_right": (0.2, 1.6),
    }

    for frame_index in range(num_frames):
        rows[frame_index, 0] = frame_index / 30.0
        horizontal_shift = frame_index * 0.01
        for joint_name, (base_x, base_y) in joint_positions.items():
            joint_index = KINECT_V1_JOINT_INDEX[joint_name]
            base_offset = 1 + (8 * joint_index)
            rows[frame_index, base_offset] = 2.0
            rows[frame_index, base_offset + 1] = base_x + horizontal_shift
            rows[frame_index, base_offset + 2] = base_y
            rows[frame_index, base_offset + 3] = 2.0
    return rows


class TestTrainingData:
    """Tests for FFD/custom data preparation helpers."""

    def test_normalize_action_label_accepts_aliases(self):
        """Test aliases normalize to canonical action codes."""
        from src.training import normalize_action_label

        assert normalize_action_label("sf") == "SF"
        assert normalize_action_label("step forward") == "SF"
        assert normalize_action_label("W/W") == "WW"
        assert normalize_action_label("jumping-sliding lunge") == "JS"

    def test_parse_clip_labels_csv_builds_records(self, tmp_path):
        """Test custom clip CSV parsing and default sample IDs."""
        from src.training import parse_clip_labels_csv

        csv_path = tmp_path / "labels.csv"
        csv_path.write_text(
            "\n".join([
                "video_path,label,start_frame,end_frame,subject_id,sample_id,notes",
                "video/clip001.mp4,step forward,3,18,athlete_001,,clean repetition",
            ]) + "\n",
            encoding="utf-8",
        )

        records = parse_clip_labels_csv(csv_path)

        assert len(records) == 1
        assert records[0].label == "SF"
        assert records[0].start_frame == 3
        assert records[0].end_frame == 18
        assert records[0].subject_id == "athlete_001"
        assert records[0].sample_id == "clip001:3:18"

    def test_prepare_ffd_dataset_converts_synthetic_body_mat(self, tmp_path):
        """Test FFD conversion yields model-ready 28x9x2 samples."""
        from src.training import prepare_ffd_dataset

        sequence_dir = tmp_path / "fencer_01" / "sf"
        sequence_dir.mkdir(parents=True, exist_ok=True)
        mat_path = sequence_dir / "fencer_01_sf_Body.mat"
        savemat(mat_path, {"body": _synthetic_ffd_body_rows(35)})

        dataset = prepare_ffd_dataset(
            dataset_root=tmp_path,
            windows_per_sequence=3,
            random_seed=0,
        )

        assert dataset.samples.shape == (3, 28, 9, 2)
        assert np.all(dataset.labels == 4)
        assert dataset.metadata[0]["subject_id"] == "fencer_01"
        assert dataset.metadata[0]["label"] == "SF"

    def test_save_and_load_prepared_dataset_roundtrip(self, tmp_path):
        """Test prepared dataset bundles round-trip through compressed NPZ."""
        from src.training import PreparedDataset, load_prepared_dataset, save_prepared_dataset

        original = PreparedDataset(
            samples=np.zeros((2, 28, 9, 2), dtype=np.float32),
            labels=np.array([0, 5], dtype=np.int64),
            action_classes=("R", "IS", "WW", "JS", "SF", "SB"),
            metadata=[
                {"sample_id": "sample_a", "subject_id": "fencer_01"},
                {"sample_id": "sample_b", "subject_id": "fencer_02"},
            ],
        )
        dataset_path = tmp_path / "prepared.npz"

        save_prepared_dataset(original, dataset_path)
        loaded = load_prepared_dataset(dataset_path)

        assert loaded.samples.shape == original.samples.shape
        assert loaded.labels.tolist() == [0, 5]
        assert loaded.metadata[1]["subject_id"] == "fencer_02"

    def test_prepare_labeled_video_dataset_uses_mock_pose(self, tmp_path):
        """Test labeled video clips can be prepared through the pose pipeline."""
        from src.training import prepare_labeled_video_dataset

        video_path = tmp_path / "clip.avi"
        _write_mock_video(video_path, frame_count=12)

        csv_path = tmp_path / "labels.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow([
                "video_path",
                "label",
                "start_frame",
                "end_frame",
                "subject_id",
                "sample_id",
                "notes",
            ])
            writer.writerow([
                str(video_path),
                "SF",
                "0",
                "12",
                "athlete_001",
                "clip_001",
                "mock clip",
            ])

        dataset = prepare_labeled_video_dataset(
            csv_path=csv_path,
            pose_backend="mock",
        )

        assert dataset.samples.shape == (1, 28, 9, 2)
        assert dataset.labels.tolist() == [4]
        assert dataset.metadata[0]["sample_id"] == "clip_001"


class TestTrainingLoop:
    """Tests for training utilities."""

    def test_split_dataset_indices_holdout_subject(self):
        """Test holdout splitting keeps one subject entirely in validation."""
        from src.training import PreparedDataset, split_dataset_indices

        dataset = PreparedDataset(
            samples=np.zeros((4, 28, 9, 2), dtype=np.float32),
            labels=np.array([0, 0, 4, 4], dtype=np.int64),
            action_classes=("R", "IS", "WW", "JS", "SF", "SB"),
            metadata=[
                {"subject_id": "fencer_01"},
                {"subject_id": "fencer_01"},
                {"subject_id": "fencer_02"},
                {"subject_id": "fencer_02"},
            ],
        )

        train_indices, validation_indices = split_dataset_indices(
            dataset=dataset,
            holdout_subject="fencer_02",
        )

        assert train_indices.tolist() == [0, 1]
        assert validation_indices.tolist() == [2, 3]

    def test_train_model_writes_best_and_last_checkpoints(self, tmp_path):
        """Test one short training run writes compatible checkpoints and metrics."""
        from src.training import (
            PreparedDataset,
            TrainingConfig,
            build_dataloaders,
            build_model,
            train_model,
        )

        rng = np.random.default_rng(0)
        samples = rng.normal(size=(8, 28, 9, 2)).astype(np.float32)
        labels = np.array([0, 0, 0, 0, 4, 4, 4, 4], dtype=np.int64)
        dataset = PreparedDataset(
            samples=samples,
            labels=labels,
            action_classes=("R", "IS", "WW", "JS", "SF", "SB"),
            metadata=[
                {"subject_id": "fencer_01", "sample_id": f"sample_{index}"}
                for index in range(8)
            ],
        )
        train_loader, validation_loader = build_dataloaders(
            dataset=dataset,
            train_indices=[0, 1, 2, 4, 5, 6],
            validation_indices=[3, 7],
            batch_size=2,
            num_workers=0,
        )
        model = build_model("fencenet", device="cpu")
        summary = train_model(
            model=model,
            train_loader=train_loader,
            validation_loader=validation_loader,
            config=TrainingConfig(
                model_type="fencenet",
                device="cpu",
                epochs=1,
                batch_size=2,
            ),
            checkpoint_dir=str(tmp_path),
            extra_checkpoint_metadata={"dataset_path": "synthetic.npz"},
        )

        assert summary["best_epoch"] == 1
        assert 0.0 <= summary["best_validation_accuracy"] <= 1.0
        assert (tmp_path / "best.pt").exists()
        assert (tmp_path / "last.pt").exists()
