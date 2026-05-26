import torch
import torch.nn as nn
from .fencenet import FENCENET_BLOCK_CONFIGS, build_fencenet_tcn_stack
from typing import List, Optional, Sequence, Tuple


class BiFenceNet(nn.Module):
    """
    BiFenceNet variant from Zhu et al. CVPRW 2022.

    Architecture:
    - One FenceNet TCN stack for forward motion
    - One separate TCN stack over reversed motion for anti-causal context
    - Concatenated last time-step features
    - Dense 64 -> Dense 6 classifier
    """

    NUM_CLASSES = 6

    def __init__(
        self,
        input_channels: int = 18,
        block_configs: Optional[Sequence[Tuple[int, int, int]]] = None,
        dropout: float = 0.5,
        device: str = "cpu"
    ):
        """
        Initialize BiFenceNet.

        Args:
            input_channels: Number of input channels (9 joints * 2 coordinates)
            block_configs: Optional (out_channels, kernel_size, dilation) stack
            dropout: Dropout rate
            device: Device to use (cpu or cuda)
        """
        super().__init__()

        self.input_channels = input_channels
        self.block_configs = tuple(block_configs or FENCENET_BLOCK_CONFIGS)
        self.device = device

        self.forward_tcn_blocks, output_channels = build_fencenet_tcn_stack(
            input_channels=input_channels,
            block_configs=self.block_configs,
            dropout=dropout
        )
        self.reverse_tcn_blocks, reverse_output_channels = build_fencenet_tcn_stack(
            input_channels=input_channels,
            block_configs=self.block_configs,
            dropout=dropout
        )
        if reverse_output_channels != output_channels:
            raise ValueError("Forward and reverse TCN stacks must match channels")

        self.fc1 = nn.Linear(2 * output_channels, 64)
        self.fc2 = nn.Linear(64, self.NUM_CLASSES)
        self.fc_activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through BiFenceNet.

        Args:
            x: Input tensor of shape (batch, channels, time_steps)
               where channels = num_joints * 2 (x, y coordinates)

        Returns:
            Logits of shape (batch, NUM_CLASSES)
        """
        forward = x
        for tcn_block in self.forward_tcn_blocks:
            forward = tcn_block(forward)
        forward_repr = forward[:, :, -1]

        reverse = torch.flip(x, dims=[2])
        for tcn_block in self.reverse_tcn_blocks:
            reverse = tcn_block(reverse)
        reverse_repr = reverse[:, :, -1]

        combined = torch.cat([forward_repr, reverse_repr], dim=1)
        out = self.fc_activation(self.fc1(combined))
        return self.fc2(out)

    def forward_with_attention(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with intermediate representations for analysis.

        Args:
            x: Input tensor of shape (batch, channels, time_steps)

        Returns:
            Tuple of (logits, bidirectional_features)
        """
        forward = x
        for tcn_block in self.forward_tcn_blocks:
            forward = tcn_block(forward)
        forward_repr = forward[:, :, -1]

        reverse = torch.flip(x, dims=[2])
        for tcn_block in self.reverse_tcn_blocks:
            reverse = tcn_block(reverse)
        reverse_repr = reverse[:, :, -1]

        combined = torch.cat([forward_repr, reverse_repr], dim=1)
        logits = self.fc2(self.fc_activation(self.fc1(combined)))
        return logits, combined

    def get_class_names(self) -> List[str]:
        """Get fencing action class names."""
        return ["R", "IS", "WW", "JS", "SF", "SB"]
