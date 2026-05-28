"""
FenceNet v2 — Spec-compliant TCN for fencing action recognition.

Architecture (from FENCENET_TRAINING.md):
  - TCNBlock with weight_norm + spatial dropout (p=0.2)
  - 6 stacked blocks with progressive channel widths and exponential dilation
  - Last time-step extraction → Dense(64, ReLU) → Dense(6, logits)

This file is self-contained so it can be imported independently of the
legacy fencenet.py / tcn_block.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


# ---------------------------------------------------------------------------
# Spatial Dropout for 1-D sequences (drops entire channels)
# ---------------------------------------------------------------------------

class SpatialDropout1d(nn.Module):
    """Drop entire channels (feature maps) instead of individual elements."""

    def __init__(self, p: float = 0.2):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, channels, time)
        """
        if not self.training or self.p == 0:
            return x
        # Create mask of shape (batch, channels, 1) and broadcast over time
        mask = x.new_empty(x.size(0), x.size(1), 1).bernoulli_(1 - self.p)
        mask = mask / (1 - self.p)           # inverted dropout scaling
        return x * mask


# ---------------------------------------------------------------------------
# TCN Block (spec-compliant)
# ---------------------------------------------------------------------------

class TCNBlockV2(nn.Module):
    """
    Single TCN block with:
      1. Dilated Causal Conv1D  →  WeightNorm  →  ReLU  →  SpatialDropout
      2. Dilated Causal Conv1D  →  WeightNorm  →  ReLU  →  SpatialDropout
      3. Residual:  output = ReLU(input + block_output)
         (1×1 conv if channels differ)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation  # left-only causal pad

        # --- Sub-block 1 ---
        self.conv1 = nn.utils.parametrizations.weight_norm(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                dilation=dilation,
            )
        )
        self.drop1 = SpatialDropout1d(dropout)

        # --- Sub-block 2 ---
        self.conv2 = nn.utils.parametrizations.weight_norm(
            nn.Conv1d(
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                dilation=dilation,
            )
        )
        self.drop2 = SpatialDropout1d(dropout)

        # --- Residual shortcut ---
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else None
        )

        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, in_channels, time)
        Returns:
            (batch, out_channels, time)
        """
        residual = x

        # Sub-block 1
        out = F.pad(x, (self.padding, 0))      # causal left-pad
        out = self.conv1(out)
        out = self.relu(out)
        out = self.drop1(out)

        # Sub-block 2
        out = F.pad(out, (self.padding, 0))
        out = self.conv2(out)
        out = self.relu(out)
        out = self.drop2(out)

        # Residual connection
        if self.downsample is not None:
            residual = self.downsample(residual)

        return self.relu(out + residual)


# ---------------------------------------------------------------------------
# FenceNet v2
# ---------------------------------------------------------------------------

class FenceNetV2(nn.Module):
    """
    Spec-compliant FenceNet.

    6 TCN blocks with progressive channel sizes and exponential dilation:
        Block 0:  18 →  32,  dilation=1
        Block 1:  32 →  32,  dilation=2
        Block 2:  32 →  64,  dilation=4
        Block 3:  64 →  64,  dilation=8
        Block 4:  64 → 128,  dilation=16
        Block 5: 128 → 128,  dilation=32

    Head:
        last_time_step → Linear(128, 64) → ReLU → Linear(64, 6)
    """

    NUM_CLASSES = 6
    CLASS_NAMES = ["R", "IS", "WW", "JS", "SF", "SB"]

    # (in_channels, out_channels) per block
    CHANNEL_PLAN = [
        (18,  32),
        (32,  32),
        (32,  64),
        (64,  64),
        (64,  128),
        (128, 128),
    ]

    def __init__(
        self,
        input_channels: int = 18,
        kernel_size: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()

        blocks = []
        for i, (c_in, c_out) in enumerate(self.CHANNEL_PLAN):
            dilation = 2 ** i
            blocks.append(
                TCNBlockV2(
                    in_channels=c_in,
                    out_channels=c_out,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
            )
        self.tcn = nn.Sequential(*blocks)

        last_ch = self.CHANNEL_PLAN[-1][1]  # 128
        self.head = nn.Sequential(
            nn.Linear(last_ch, 64),
            nn.ReLU(),
            nn.Linear(64, self.NUM_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 18, T)  where T >= 28
        Returns:
            logits: (batch, 6)
        """
        out = self.tcn(x)                    # (batch, 128, T)
        out = out[:, :, -1]                  # last time-step → (batch, 128)
        out = self.head(out)                 # (batch, 6)
        return out

    def get_class_names(self) -> List[str]:
        return list(self.CLASS_NAMES)
