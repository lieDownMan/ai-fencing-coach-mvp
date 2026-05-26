import torch
import torch.nn as nn
from .tcn_block import TCNBlock
from typing import List, Optional, Sequence, Tuple


FENCENET_BLOCK_CONFIGS: Tuple[Tuple[int, int, int], ...] = (
    # (out_channels, kernel_size, dilation), matching Fig. 1a of Zhu et al.
    # The paper has one sentence saying 6 blocks; Fig. 1a shows these 4.
    (32, 8, 1),
    (64, 8, 2),
    (64, 7, 4),
    (128, 5, 6),
)


def build_fencenet_tcn_stack(
    input_channels: int,
    block_configs: Sequence[Tuple[int, int, int]],
    dropout: float
) -> Tuple[nn.ModuleList, int]:
    """Build the tuned FenceNet TCN stack and return its output channels."""
    blocks = nn.ModuleList()
    current_channels = input_channels
    for out_channels, kernel_size, dilation in block_configs:
        blocks.append(
            TCNBlock(
                in_channels=current_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                dilation=dilation,
                dropout=dropout
            )
        )
        current_channels = out_channels
    return blocks, current_channels


class FenceNet(nn.Module):
    """
    FenceNet from "Fine-Grained Footwork Recognition in Fencing".

    Architecture:
    - 18-channel input: 9 2D joints after nose/front-ankle normalization
    - Tuned TCN stack from paper Figure 1a
    - Last time-step extraction
    - Dense 64 -> Dense 6 classifier

    Target classes:
    1. R: Rapid lunge
    2. IS: Incremental speed lunge
    3. WW: With waiting lunge
    4. JS: Jumping sliding lunge
    5. SF: Step forward
    6. SB: Step backward
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
        Initialize FenceNet.

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

        self.tcn_blocks, tcn_output_channels = build_fencenet_tcn_stack(
            input_channels=input_channels,
            block_configs=self.block_configs,
            dropout=dropout
        )

        self.fc1 = nn.Linear(tcn_output_channels, 64)
        self.fc2 = nn.Linear(64, self.NUM_CLASSES)
        self.fc_activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through FenceNet.

        Args:
            x: Input tensor of shape (batch, channels, time_steps)
               where channels = num_joints * 2 (x, y coordinates)

        Returns:
            Logits of shape (batch, NUM_CLASSES)
        """
        out = x
        for tcn_block in self.tcn_blocks:
            out = tcn_block(out)

        out = out[:, :, -1]
        out = self.fc_activation(self.fc1(out))
        return self.fc2(out)

    def get_class_names(self) -> List[str]:
        """Get fencing action class names."""
        return ["R", "IS", "WW", "JS", "SF", "SB"]
