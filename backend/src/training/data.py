"""Dataset preparation helpers for FenceNet/BiFenceNet training."""

from __future__ import annotations

import csv
import json
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from scipy.io import loadmat

from ..pose_estimation import PoseEstimator
from ..preprocessing import SpatialNormalizer, TemporalSampler

logger = logging.getLogger(__name__)

ACTION_CLASSES: Tuple[str, ...] = ("R", "IS", "WW", "JS", "SF", "SB")
ACTION_TO_INDEX = {label: index for index, label in enumerate(ACTION_CLASSES)}
WINDOW_SIZE = 28
FFD_MAX_RANDOM_START = 20
FFD_WINDOWS_PER_SEQUENCE = 10

# Kinect v1 joint order, matching the public FFD documentation.
KINECT_V1_JOINT_INDEX = {
    "hip_center": 0,
    "spine": 1,
    "shoulder_center": 2,
    "head": 3,
    "shoulder_left": 4,
    "elbow_left": 5,
    "wrist_left": 6,
    "hand_left": 7,
    "shoulder_right": 8,
    "elbow_right": 9,
    "wrist_right": 10,
    "hand_right": 11,
    "hip_left": 12,
    "knee_left": 13,
    "ankle_left": 14,
    "foot_left": 15,
    "hip_right": 16,
    "knee_right": 17,
    "ankle_right": 18,
    "foot_right": 19,
}

# FFD stores a head joint rather than a nose joint, so we use head as the
# closest available surrogate for the paper's normalization anchor.
KINECT_TO_SKELETON_JOINT = {
    "nose": "head",
    "front_wrist": "wrist_right",
    "front_elbow": "elbow_right",
    "front_shoulder": "shoulder_right",
    "left_hip": "hip_left",
    "right_hip": "hip_right",
    "left_knee": "knee_left",
    "right_knee": "knee_right",
    "left_ankle": "ankle_left",
    "right_ankle": "ankle_right",
    "front_ankle": "ankle_right",
}

_ACTION_ALIASES = {
    "r": "R",
    "rapid": "R",
    "rapidlunge": "R",
    "is": "IS",
    "incrementalspeed": "IS",
    "incrementalspeedlunge": "IS",
    "ww": "WW",
    "withwaiting": "WW",
    "lungewithwaiting": "WW",
    "js": "JS",
    "jumpingsliding": "JS",
    "jumpingslidinglunge": "JS",
    "sf": "SF",
    "stepforward": "SF",
    "sb": "SB",
    "stepbackward": "SB",
}


@dataclass
class ClipLabelRecord:
    """One user-labeled action clip."""

    video_path: str
    label: str
    start_frame: int = 0
    end_frame: Optional[int] = None
    subject_id: str = "unknown"
    sample_id: str = ""
    notes: str = ""


@dataclass
class PreparedDataset:
    """Serializable prepared training dataset."""

    samples: np.ndarray
    labels: np.ndarray
    action_classes: Tuple[str, ...]
    metadata: List[Dict[str, Any]]

    def summary(self) -> Dict[str, Any]:
        """Return JSON-friendly dataset summary."""
        label_counts = {
            action_label: int(np.sum(self.labels == action_index))
            for action_index, action_label in enumerate(self.action_classes)
        }
        return {
            "num_samples": int(self.samples.shape[0]),
            "window_size": int(self.samples.shape[1]) if self.samples.ndim >= 2 else 0,
            "num_joints": int(self.samples.shape[2]) if self.samples.ndim >= 3 else 0,
            "num_channels": int(self.samples.shape[2] * self.samples.shape[3])
            if self.samples.ndim == 4
            else 0,
            "label_counts": label_counts,
        }


def normalize_action_label(label: str) -> str:
    """Normalize action label aliases to one canonical FenceNet class code."""
    if not isinstance(label, str) or not label.strip():
        raise ValueError("Action label must be a non-empty string")

    normalized = re.sub(r"[^a-z0-9]+", "", label.strip().lower())
    if normalized in _ACTION_ALIASES:
        return _ACTION_ALIASES[normalized]
    raise ValueError(
        f"Unsupported action label '{label}'. Expected one of {ACTION_CLASSES} "
        "or a known alias."
    )


def infer_action_label_from_path(path: Path) -> str:
    """Infer an action label from a dataset path."""
    for part in reversed(path.parts):
        part_text = part.lower()
        tokens = [token for token in re.split(r"[^a-z0-9]+", part_text) if token]
        candidates = list(tokens)
        candidates.append("".join(tokens))
        for ngram_size in (2, 3):
            for index in range(0, len(tokens) - ngram_size + 1):
                candidates.append("".join(tokens[index:index + ngram_size]))

        for candidate in candidates:
            if candidate in _ACTION_ALIASES:
                return _ACTION_ALIASES[candidate]

    raise ValueError(f"Could not infer action label from path: {path}")


def infer_subject_id_from_path(
    path: Path,
    dataset_root: Optional[Path] = None
) -> str:
    """Infer a subject/fencer identifier from a dataset path."""
    relative_parts = (
        path.relative_to(dataset_root).parts
        if dataset_root is not None and path.is_relative_to(dataset_root)
        else path.parts
    )

    action_like_parts = set()
    for part in relative_parts:
        try:
            action_like_parts.add(infer_action_label_from_path(Path(part)))
        except ValueError:
            continue

    subject_patterns = [
        re.compile(r"^(?:fencer|person|subject|athlete)[_-]?(\d{1,3})$", re.I),
        re.compile(r"^(?:p|s)[_-]?(\d{1,3})$", re.I),
    ]

    for part in reversed(relative_parts[:-1]):
        stripped = part.strip()
        for pattern in subject_patterns:
            match = pattern.match(stripped)
            if match:
                return f"fencer_{int(match.group(1)):02d}"

        try:
            normalized = infer_action_label_from_path(Path(stripped))
        except ValueError:
            normalized = None
        if normalized in action_like_parts:
            continue

        if re.fullmatch(r"\d{1,3}", stripped):
            numeric_value = int(stripped)
            if numeric_value <= 99:
                return f"fencer_{numeric_value:02d}"

        if stripped and not re.fullmatch(r"\d{4}(-\d{2}){2}", stripped):
            return re.sub(r"[^A-Za-z0-9._-]+", "_", stripped)

    return "unknown"


def load_ffd_body_rows(mat_path: Path) -> np.ndarray:
    """Load FFD Kinect body rows from a MATLAB file."""
    matlab_data = loadmat(mat_path)
    candidate_arrays = [
        value
        for key, value in matlab_data.items()
        if not key.startswith("__") and isinstance(value, np.ndarray) and value.ndim == 2
    ]
    if not candidate_arrays:
        raise ValueError(f"No 2D matrix found in MATLAB file: {mat_path}")

    body_rows = max(candidate_arrays, key=lambda value: value.shape[0] * value.shape[1])
    if body_rows.shape[1] == 161:
        return np.asarray(body_rows, dtype=float)
    if body_rows.shape[0] == 161:
        return np.asarray(body_rows.T, dtype=float)

    raise ValueError(
        f"Unexpected FFD body matrix shape {body_rows.shape} in {mat_path}; "
        "expected n x 161."
    )


def _kinect_camera_xy(row: np.ndarray, joint_name: str) -> Tuple[float, float]:
    """Extract one joint in Kinect camera x/y space from an FFD body row."""
    joint_index = KINECT_V1_JOINT_INDEX[joint_name]
    base_index = 1 + (8 * joint_index)
    tracking_status = float(row[base_index])
    x_coord = float(row[base_index + 1])
    y_coord = float(row[base_index + 2])

    if tracking_status <= 0:
        raise ValueError(f"Joint '{joint_name}' is not tracked")
    if not np.isfinite(x_coord) or not np.isfinite(y_coord):
        raise ValueError(f"Joint '{joint_name}' contains non-finite coordinates")

    return x_coord, y_coord


def body_row_to_skeleton(row: np.ndarray) -> Dict[str, Tuple[float, float]]:
    """Convert one FFD body-row frame into this repo's skeleton schema."""
    if row.shape[0] < 161:
        raise ValueError("FFD body row must contain at least 161 values")

    skeleton = {}
    for output_joint, kinect_joint in KINECT_TO_SKELETON_JOINT.items():
        skeleton[output_joint] = _kinect_camera_xy(row, kinect_joint)
    return skeleton


def normalize_skeleton_sequence(
    skeleton_sequence: Sequence[Dict[str, Tuple[float, float]]],
    target_length: int = WINDOW_SIZE,
    resample: bool = True
) -> np.ndarray:
    """Normalize a skeleton sequence and export a model-ready array."""
    if not skeleton_sequence:
        raise ValueError("skeleton_sequence must contain at least one frame")

    normalizer = SpatialNormalizer()
    normalizer.fit(list(skeleton_sequence))
    normalized_array = normalizer.get_normalized_array(
        list(skeleton_sequence),
        joint_names=list(SpatialNormalizer.MODEL_JOINT_NAMES)
    )

    if target_length is not None and normalized_array.shape[0] != target_length:
        if not resample:
            raise ValueError(
                "Sequence length does not match target_length and resample=False"
            )
        sampler = TemporalSampler(target_length=target_length)
        normalized_array = sampler.sample_array(normalized_array)

    return normalized_array.astype(np.float32)


def prepare_ffd_dataset(
    dataset_root: Path,
    window_size: int = WINDOW_SIZE,
    windows_per_sequence: int = FFD_WINDOWS_PER_SEQUENCE,
    max_random_start: int = FFD_MAX_RANDOM_START,
    random_seed: int = 42
) -> PreparedDataset:
    """Convert the public FFD dataset into model-ready training windows."""
    dataset_root = Path(dataset_root).expanduser()
    if not dataset_root.exists():
        placeholder_hint = ""
        if str(dataset_root) == "/path/to/ffd":
            placeholder_hint = (
                " The value '/path/to/ffd' is only a placeholder from the docs. "
                "Replace it with the real directory where you unpacked FFD."
            )
        raise FileNotFoundError(
            f"FFD dataset root not found: {dataset_root}.{placeholder_hint}"
        )
    if windows_per_sequence <= 0:
        raise ValueError("windows_per_sequence must be positive")
    if max_random_start < 0:
        raise ValueError("max_random_start must be non-negative")

    mat_paths = sorted(dataset_root.rglob("*_Body.mat"))
    if not mat_paths:
        raise FileNotFoundError(
            "No FFD *_Body.mat files found under "
            f"{dataset_root}. Point --ffd-root at the unpacked FFD directory "
            "or one of its parent folders."
        )

    rng = np.random.default_rng(random_seed)
    samples: List[np.ndarray] = []
    labels: List[int] = []
    metadata: List[Dict[str, Any]] = []

    for mat_path in mat_paths:
        action_label = infer_action_label_from_path(mat_path)
        subject_id = infer_subject_id_from_path(mat_path, dataset_root=dataset_root)
        body_rows = load_ffd_body_rows(mat_path)

        skeleton_frames = []
        for row_index, row in enumerate(body_rows):
            try:
                skeleton_frames.append(body_row_to_skeleton(np.asarray(row, dtype=float)))
            except ValueError as exc:
                logger.debug("Skipping invalid FFD frame %s:%s (%s)", mat_path, row_index, exc)

        if len(skeleton_frames) < window_size:
            logger.warning(
                "Skipping %s because it has only %s valid skeleton frames",
                mat_path,
                len(skeleton_frames)
            )
            continue

        max_start_index = min(max_random_start, len(skeleton_frames) - window_size)
        candidate_starts = list(range(0, max_start_index + 1))
        sample_count = min(windows_per_sequence, len(candidate_starts))
        if sample_count == len(candidate_starts):
            selected_starts = candidate_starts
        else:
            selected_starts = sorted(
                int(value)
                for value in rng.choice(candidate_starts, size=sample_count, replace=False)
            )

        for start_frame in selected_starts:
            end_frame = start_frame + window_size
            window = skeleton_frames[start_frame:end_frame]
            samples.append(
                normalize_skeleton_sequence(
                    window,
                    target_length=window_size,
                    resample=False
                )
            )
            labels.append(ACTION_TO_INDEX[action_label])
            metadata.append({
                "dataset_type": "ffd",
                "sample_id": f"{mat_path.stem}:{start_frame}",
                "source_path": str(mat_path),
                "label": action_label,
                "subject_id": subject_id,
                "start_frame": int(start_frame),
                "end_frame_exclusive": int(end_frame),
                "valid_frame_count": int(len(skeleton_frames)),
            })

    if not samples:
        raise ValueError(
            f"No valid FFD samples were prepared from {dataset_root}"
        )

    return PreparedDataset(
        samples=np.stack(samples, axis=0).astype(np.float32),
        labels=np.asarray(labels, dtype=np.int64),
        action_classes=ACTION_CLASSES,
        metadata=metadata,
    )


def parse_clip_labels_csv(csv_path: Path) -> List[ClipLabelRecord]:
    """Parse a simple clip-label CSV for custom training clips."""
    csv_path = Path(csv_path).expanduser()
    if not csv_path.exists():
        raise FileNotFoundError(f"Clip label CSV not found: {csv_path}")

    with csv_path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            raise ValueError("Clip label CSV must include a header row")
        required_fields = {"video_path", "label"}
        missing_fields = required_fields - set(reader.fieldnames)
        if missing_fields:
            raise ValueError(
                f"Clip label CSV is missing required columns: {sorted(missing_fields)}"
            )

        records = []
        for row_index, row in enumerate(reader, start=2):
            video_path = (row.get("video_path") or "").strip()
            label = normalize_action_label(row.get("label") or "")
            if not video_path:
                raise ValueError(f"Row {row_index} is missing video_path")

            start_frame = _parse_optional_int(row.get("start_frame"), default=0)
            end_frame = _parse_optional_int(row.get("end_frame"), default=None)
            if start_frame < 0:
                raise ValueError(f"Row {row_index} start_frame must be >= 0")
            if end_frame is not None and end_frame <= start_frame:
                raise ValueError(
                    f"Row {row_index} end_frame must be greater than start_frame"
                )

            sample_id = (row.get("sample_id") or "").strip()
            if not sample_id:
                clip_name = Path(video_path).stem
                clip_end = "end" if end_frame is None else str(end_frame)
                sample_id = f"{clip_name}:{start_frame}:{clip_end}"

            records.append(
                ClipLabelRecord(
                    video_path=video_path,
                    label=label,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    subject_id=(row.get("subject_id") or "unknown").strip() or "unknown",
                    sample_id=sample_id,
                    notes=(row.get("notes") or "").strip(),
                )
            )

    if not records:
        raise ValueError(f"No rows found in clip label CSV: {csv_path}")
    return records


def write_clip_label_template(output_path: Path):
    """Write a starter CSV template for custom labeled clips."""
    output_path = Path(output_path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="") as csv_file:
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
            "video/my_session_clip_001.mp4",
            "SF",
            "0",
            "42",
            "athlete_001",
            "clip_001",
            "single clean step-forward repetition",
        ])


def extract_labeled_clip_skeletons(
    record: ClipLabelRecord,
    pose_estimator: PoseEstimator
) -> List[Dict[str, Tuple[float, float]]]:
    """Extract one labeled skeleton sequence from a video clip definition."""
    video_path = Path(record.video_path).expanduser()
    if not video_path.exists():
        raise FileNotFoundError(f"Labeled clip video not found: {video_path}")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Cannot open labeled clip video: {video_path}")

    try:
        current_frame = 0
        skeletons: List[Dict[str, Tuple[float, float]]] = []
        if record.start_frame:
            capture.set(cv2.CAP_PROP_POS_FRAMES, float(record.start_frame))
            current_frame = int(record.start_frame)

        while True:
            if record.end_frame is not None and current_frame >= record.end_frame:
                break

            ret, frame = capture.read()
            if not ret:
                break

            skeleton = pose_estimator.extract_frame_skeleton(frame)
            if skeleton is not None and pose_estimator.validate_skeleton(skeleton):
                skeletons.append(skeleton)
            current_frame += 1

        if not skeletons:
            raise ValueError(
                f"No valid skeleton frames were extracted for labeled clip {record.sample_id}"
            )
        return skeletons
    finally:
        capture.release()


def prepare_labeled_video_dataset(
    csv_path: Path,
    pose_backend: str = "ultralytics",
    pose_model_path: Optional[str] = None,
    window_size: int = WINDOW_SIZE
) -> PreparedDataset:
    """Prepare a model-ready dataset from a simple labeled-clip CSV."""
    records = parse_clip_labels_csv(csv_path)
    pose_estimator = PoseEstimator(model_path=pose_model_path, backend=pose_backend)

    samples: List[np.ndarray] = []
    labels: List[int] = []
    metadata: List[Dict[str, Any]] = []

    for record in records:
        skeletons = extract_labeled_clip_skeletons(record, pose_estimator)
        samples.append(
            normalize_skeleton_sequence(
                skeletons,
                target_length=window_size,
                resample=True
            )
        )
        labels.append(ACTION_TO_INDEX[record.label])
        metadata.append({
            "dataset_type": "labeled_video",
            "sample_id": record.sample_id,
            "source_path": str(Path(record.video_path).expanduser()),
            "label": record.label,
            "subject_id": record.subject_id,
            "start_frame": int(record.start_frame),
            "end_frame_exclusive": (
                int(record.end_frame) if record.end_frame is not None else None
            ),
            "notes": record.notes,
            "valid_frame_count": int(len(skeletons)),
        })

    return PreparedDataset(
        samples=np.stack(samples, axis=0).astype(np.float32),
        labels=np.asarray(labels, dtype=np.int64),
        action_classes=ACTION_CLASSES,
        metadata=metadata,
    )


def save_prepared_dataset(dataset: PreparedDataset, output_path: Path):
    """Save a prepared dataset as a compressed NPZ bundle."""
    output_path = Path(output_path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        samples=dataset.samples.astype(np.float32),
        labels=dataset.labels.astype(np.int64),
        action_classes=np.asarray(dataset.action_classes, dtype="<U8"),
        metadata_json=np.asarray(json.dumps(dataset.metadata, indent=2)),
    )


def load_prepared_dataset(dataset_path: Path) -> PreparedDataset:
    """Load a prepared dataset from a compressed NPZ bundle."""
    dataset_path = Path(dataset_path).expanduser()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Prepared dataset not found: {dataset_path}")

    bundle = np.load(dataset_path, allow_pickle=False)
    samples = np.asarray(bundle["samples"], dtype=np.float32)
    labels = np.asarray(bundle["labels"], dtype=np.int64)
    action_classes = tuple(str(value) for value in bundle["action_classes"].tolist())
    metadata = json.loads(str(bundle["metadata_json"].item()))

    if samples.ndim != 4 or samples.shape[1:] != (WINDOW_SIZE, 9, 2):
        raise ValueError(
            "Prepared dataset samples must have shape (N, 28, 9, 2)"
        )
    if labels.ndim != 1 or labels.shape[0] != samples.shape[0]:
        raise ValueError("Prepared dataset labels must align with samples")
    if len(metadata) != samples.shape[0]:
        raise ValueError("Prepared dataset metadata must align with samples")

    return PreparedDataset(
        samples=samples,
        labels=labels,
        action_classes=action_classes,
        metadata=metadata,
    )


def split_dataset_indices(
    dataset: PreparedDataset,
    validation_ratio: float = 0.2,
    random_seed: int = 42,
    holdout_subject: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Split a prepared dataset into training and validation indices."""
    total_samples = dataset.samples.shape[0]
    if total_samples < 2:
        raise ValueError("Prepared dataset must contain at least two samples")

    if holdout_subject is not None:
        holdout_subject = holdout_subject.strip()
        train_indices = []
        validation_indices = []
        for index, item in enumerate(dataset.metadata):
            if str(item.get("subject_id", "")).strip() == holdout_subject:
                validation_indices.append(index)
            else:
                train_indices.append(index)
        if not validation_indices:
            raise ValueError(
                f"Holdout subject '{holdout_subject}' was not found in the dataset"
            )
        if not train_indices:
            raise ValueError(
                f"Holdout subject '{holdout_subject}' leaves no training samples"
            )
        return (
            np.asarray(train_indices, dtype=np.int64),
            np.asarray(validation_indices, dtype=np.int64),
        )

    if not 0.0 < validation_ratio < 1.0:
        raise ValueError("validation_ratio must be between 0 and 1")

    rng = np.random.default_rng(random_seed)
    indices = np.arange(total_samples, dtype=np.int64)
    rng.shuffle(indices)

    validation_size = max(1, int(round(total_samples * validation_ratio)))
    validation_size = min(validation_size, total_samples - 1)
    validation_indices = np.sort(indices[:validation_size])
    train_indices = np.sort(indices[validation_size:])

    return train_indices, validation_indices


def dataset_metadata_rows(dataset: PreparedDataset) -> List[Dict[str, Any]]:
    """Return metadata rows enriched with decoded labels."""
    rows = []
    for label_index, metadata in zip(dataset.labels.tolist(), dataset.metadata):
        row = dict(metadata)
        row["label_index"] = int(label_index)
        row["label_name"] = dataset.action_classes[int(label_index)]
        rows.append(row)
    return rows


def clip_label_record_to_dict(record: ClipLabelRecord) -> Dict[str, Any]:
    """Return a JSON-friendly dict for a clip-label record."""
    return asdict(record)


def _parse_optional_int(value: Optional[str], default: Optional[int]) -> Optional[int]:
    """Parse an optional integer field from CSV input."""
    if value is None:
        return default
    stripped = str(value).strip()
    if not stripped:
        return default
    return int(stripped)
