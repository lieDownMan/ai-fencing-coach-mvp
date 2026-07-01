"""
FencingDataset - PyTorch Dataset for 2D pose-based fencing action recognition.

Expected raw data format (JSON per video clip):
{
    "label": "R",                  # one of: R, IS, WW, JS, SF, SB
    "fencer_id": "fencer_01",      # used for leave-one-person-out CV
    "keypoints": [                 # list of frames (>= 48 frames recommended)
        {
            "nose":            [x, y],
            "front_wrist":     [x, y],
            "front_elbow":     [x, y],
            "front_shoulder":  [x, y],
            "left_hip":        [x, y],
            "right_hip":       [x, y],
            "left_knee":       [x, y],
            "right_knee":      [x, y],
            "left_ankle":      [x, y],
            "right_ankle":     [x, y]
        },
        ...
    ]
}

One JSON file per video clip, all placed under a single directory.
"""

import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CLASS_NAMES = ["R", "IS", "WW", "JS", "SF", "SB"]
CLASS_TO_IDX: Dict[str, int] = {name: idx for idx, name in enumerate(CLASS_NAMES)}
NUM_CLASSES = len(CLASS_NAMES)

# 9 model joints (nose is used for normalization only, not as a feature)
MODEL_JOINTS = [
    "front_wrist",
    "front_elbow",
    "front_shoulder",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]
NUM_JOINTS = len(MODEL_JOINTS)      # 9
NUM_CHANNELS = NUM_JOINTS * 2       # 18

SEQUENCE_LENGTH = 28
MAX_START_FRAME = 20                 # random crop starts in [0, MAX_START_FRAME]


# ---------------------------------------------------------------------------
# Helper: load & parse a single JSON sample
# ---------------------------------------------------------------------------

def _load_sample(path: str) -> dict:
    """Load a single JSON sample and return the parsed dict."""
    with open(path, "r") as f:
        data = json.load(f)

    required_keys = {"label", "fencer_id", "keypoints"}
    missing = required_keys - set(data.keys())
    if missing:
        raise KeyError(f"{path}: missing keys {missing}")
    if data["label"] not in CLASS_TO_IDX:
        raise ValueError(f"{path}: unknown label '{data['label']}'")

    return data


# ---------------------------------------------------------------------------
# Helper: spatial normalization
# ---------------------------------------------------------------------------

def _spatial_normalize(keypoints: List[dict]) -> np.ndarray:
    """
    Normalize keypoints according to the spec:
      1. Subtract nose(t=0) from ALL joint coords in every frame.
      2. Divide by |nose_y(t=0) - front_ankle_y(t=0)|.

    Args:
        keypoints: raw keypoints list (each element is a dict of joint->[x,y]).

    Returns:
        np.ndarray of shape (num_frames, 9, 2) — only 9 model joints.
    """
    first = keypoints[0]
    nose_x0, nose_y0 = first["nose"]
    ankle_x0, ankle_y0 = first["front_ankle"] if "front_ankle" in first else first["left_ankle"]
    scale = abs(ankle_y0 - nose_y0)
    if scale < 1e-6:
        scale = 1.0  # safety

    num_frames = len(keypoints)
    array = np.zeros((num_frames, NUM_JOINTS, 2), dtype=np.float32)

    for t, frame in enumerate(keypoints):
        for j, joint_name in enumerate(MODEL_JOINTS):
            x, y = frame[joint_name]
            array[t, j, 0] = (x - nose_x0) / scale
            array[t, j, 1] = (y - nose_y0) / scale

    return array


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class FencingDataset(Dataset):
    """
    PyTorch Dataset that yields (features, label) for FenceNet training.

    * features: FloatTensor of shape (18, 28)   — (channels, time)
    * label:    LongTensor scalar               — class index in [0, 5]

    During **training** mode (`is_train=True`):
        Randomly samples 28 consecutive frames starting from a random offset
        in [0, MAX_START_FRAME].

    During **validation / inference** mode (`is_train=False`):
        Returns ALL non-overlapping 28-frame subsequences for majority voting.
        In this mode, __getitem__ returns (features_list, label) where
        features_list has shape (N, 18, 28) and N is the number of
        subsequences.  A custom collate_fn (provided below) must be used.
    """

    def __init__(
        self,
        data_dir: str,
        sample_indices: Optional[List[int]] = None,
        is_train: bool = True,
    ):
        """
        Args:
            data_dir:       Path to directory containing JSON sample files.
            sample_indices: If given, only use these indices into the sorted
                            file list (for cross-validation splits).
            is_train:       Training mode (random crop) vs. eval mode (all subseqs).
        """
        self.data_dir = Path(data_dir)
        self.is_train = is_train

        # Discover all JSON files
        all_files = sorted(
            [f for f in os.listdir(self.data_dir) if f.endswith(".json")]
        )
        if sample_indices is not None:
            all_files = [all_files[i] for i in sample_indices]

        # Pre-load metadata (label, fencer_id, # frames) for fast access
        self.samples: List[dict] = []
        for fname in all_files:
            fpath = str(self.data_dir / fname)
            data = _load_sample(fpath)
            self.samples.append(
                {
                    "path": fpath,
                    "label": CLASS_TO_IDX[data["label"]],
                    "fencer_id": data["fencer_id"],
                    "num_frames": len(data["keypoints"]),
                }
            )

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def get_fencer_ids(self) -> List[str]:
        """Return a list of fencer IDs aligned with sample indices."""
        return [s["fencer_id"] for s in self.samples]

    def get_unique_fencer_ids(self) -> List[str]:
        """Return sorted unique fencer IDs."""
        return sorted(set(self.get_fencer_ids()))

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        meta = self.samples[idx]
        data = _load_sample(meta["path"])
        keypoints = data["keypoints"]
        label = meta["label"]

        # Spatial normalization → (num_frames, 9, 2)
        normed = _spatial_normalize(keypoints)

        if self.is_train:
            features = self._train_crop(normed)          # (18, 28)
            return features, torch.tensor(label, dtype=torch.long)
        else:
            subseqs = self._eval_subsequences(normed)     # (N, 18, 28)
            return subseqs, torch.tensor(label, dtype=torch.long)

    # ------------------------------------------------------------------
    # Temporal sampling
    # ------------------------------------------------------------------

    def _train_crop(self, normed: np.ndarray) -> torch.FloatTensor:
        """
        Randomly crop 28 frames. Start from a random offset in
        [0, min(MAX_START_FRAME, num_frames - SEQUENCE_LENGTH)].
        """
        num_frames = normed.shape[0]
        max_start = min(MAX_START_FRAME, num_frames - SEQUENCE_LENGTH)
        max_start = max(0, max_start)
        start = random.randint(0, max_start)
        clip = normed[start: start + SEQUENCE_LENGTH]  # (28, 9, 2)

        # Reshape to (channels=18, time=28)
        # From (T, J, 2) → flatten J*2 → (T, 18) → transpose → (18, T)
        clip = clip.reshape(SEQUENCE_LENGTH, NUM_CHANNELS)  # (28, 18)
        clip = clip.T  # (18, 28)
        return torch.FloatTensor(clip)

    def _eval_subsequences(self, normed: np.ndarray) -> torch.FloatTensor:
        """
        Extract ALL non-overlapping 28-frame subsequences for majority voting.
        If fewer than 28 frames, pad with the last frame.
        """
        num_frames = normed.shape[0]

        # Pad if necessary
        if num_frames < SEQUENCE_LENGTH:
            pad_count = SEQUENCE_LENGTH - num_frames
            padding = np.tile(normed[-1:], (pad_count, 1, 1))
            normed = np.concatenate([normed, padding], axis=0)
            num_frames = SEQUENCE_LENGTH

        subseqs = []
        start = 0
        while start + SEQUENCE_LENGTH <= num_frames:
            clip = normed[start: start + SEQUENCE_LENGTH]
            clip = clip.reshape(SEQUENCE_LENGTH, NUM_CHANNELS).T  # (18, 28)
            subseqs.append(clip)
            start += SEQUENCE_LENGTH

        # If there are leftover frames, take the last 28 frames as well
        if start < num_frames and start != num_frames - SEQUENCE_LENGTH + SEQUENCE_LENGTH:
            clip = normed[num_frames - SEQUENCE_LENGTH: num_frames]
            clip = clip.reshape(SEQUENCE_LENGTH, NUM_CHANNELS).T
            # Avoid duplicating if we already captured this exact window
            if len(subseqs) == 0 or not np.array_equal(subseqs[-1], clip):
                subseqs.append(clip)

        tensor = np.stack(subseqs, axis=0)  # (N, 18, 28)
        return torch.FloatTensor(tensor)


# ---------------------------------------------------------------------------
# Custom collate function for evaluation mode
# ---------------------------------------------------------------------------

def eval_collate_fn(batch):
    """
    Collate for evaluation mode where each sample has variable-length subseqs.

    Returns:
        subseqs_list: list of tensors, each (N_i, 18, 28)
        labels:       LongTensor of shape (batch_size,)
    """
    subseqs_list = [item[0] for item in batch]
    labels = torch.stack([item[1] for item in batch])
    return subseqs_list, labels
