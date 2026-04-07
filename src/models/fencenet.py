"""
FenceNet - Stack of 6 TCN blocks for fencing footwork classification.
"""

import torch
import torch.nn as nn
from .tcn_block import TCNBlock
from typing import List


class FenceNet(nn.Module):
    """
    FenceNet: Temporal Convolutional Network for fencing footwork recognition.
    
    Architecture:
    - 6 stacked TCN blocks with increasing dilation
    - Extract last time-step for final prediction
    - Dense layers for classification into 6 fencing actions
    
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
        input_channels: int = 20,  # 10 joints * 2 coordinates
        hidden_channels: int = 64,
        num_tcn_blocks: int = 6,
        kernel_size: int = 3,
        dropout: float = 0.5,
        device: str = "cpu"
    ):
        """
        Initialize FenceNet.
        
        Args:
            input_channels: Number of input channels (joints * 2)
            hidden_channels: Number of hidden channels in TCN blocks
            num_tcn_blocks: Number of TCN blocks to stack
            kernel_size: Kernel size for convolutions
            dropout: Dropout rate
            device: Device to use (cpu or cuda)
        """
        super().__init__()
        
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.num_tcn_blocks = num_tcn_blocks
        self.device = device
        
        # Activation function
        activation = nn.ReLU()
        
        # Initial projection layer
        self.input_projection = nn.Conv1d(input_channels, hidden_channels, kernel_size=1)
        
        # Stack TCN blocks with increasing dilation
        self.tcn_blocks = nn.ModuleList()
        for i in range(num_tcn_blocks):
            dilation = 2 ** i  # Exponentially increasing dilation
            self.tcn_blocks.append(
                TCNBlock(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                    activation=activation
                )
            )
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Dense layers for classification
        self.fc1 = nn.Linear(hidden_channels, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, self.NUM_CLASSES)
        
        # Dropout for dense layers
        self.fc_dropout = nn.Dropout(dropout)
        
        # Activation for dense layers
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
        # Project input to hidden channels
        out = self.input_projection(x)  # (batch, hidden_channels, time_steps)
        
        # Pass through TCN blocks
        for tcn_block in self.tcn_blocks:
            out = tcn_block(out)  # (batch, hidden_channels, time_steps)
        
        # Extract last time-step
        # out = out[:, :, -1]  # (batch, hidden_channels)
        
        # Or use global average pooling
        out = self.global_pool(out).squeeze(-1)  # (batch, hidden_channels)
        
        # Dense layers
        out = self.fc1(out)  # (batch, 128)
        out = self.fc_activation(out)
        out = self.fc_dropout(out)
        
        out = self.fc2(out)  # (batch, 64)
        out = self.fc_activation(out)
        out = self.fc_dropout(out)
        
        out = self.fc3(out)  # (batch, NUM_CLASSES)
        
        return out
    
    def get_class_names(self) -> List[str]:
        """Get fencing action class names."""
        return ["R", "IS", "WW", "JS", "SF", "SB"]
