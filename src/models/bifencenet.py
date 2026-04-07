"""
BiFenceNet - Bidirectional FenceNet with forward and reverse motion analysis.
"""

import torch
import torch.nn as nn
from .tcn_block import TCNBlock
from typing import List, Tuple


class BiFenceNet(nn.Module):
    """
    BiFenceNet: Bidirectional Temporal Convolutional Network for fencing footwork recognition.
    
    Architecture:
    - Forward TCN module: Processes motion forward in time
    - Anti-causal (reverse) TCN module: Processes motion backward in time
    - Concatenates last time-steps from both modules
    - Dense layers for final classification
    
    This allows the model to capture both forward-looking and backward-looking
    temporal patterns in fencing movements.
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
        Initialize BiFenceNet.
        
        Args:
            input_channels: Number of input channels (joints * 2)
            hidden_channels: Number of hidden channels in TCN blocks
            num_tcn_blocks: Number of TCN blocks per direction
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
        
        # Forward direction TCN blocks
        self.forward_projection = nn.Conv1d(input_channels, hidden_channels, kernel_size=1)
        self.forward_tcn_blocks = nn.ModuleList()
        for i in range(num_tcn_blocks):
            dilation = 2 ** i
            self.forward_tcn_blocks.append(
                TCNBlock(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                    activation=activation
                )
            )
        
        # Reverse direction TCN blocks (anti-causal)
        self.reverse_projection = nn.Conv1d(input_channels, hidden_channels, kernel_size=1)
        self.reverse_tcn_blocks = nn.ModuleList()
        for i in range(num_tcn_blocks):
            dilation = 2 ** i
            self.reverse_tcn_blocks.append(
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
        # Input is concatenation of forward and reverse: 2 * hidden_channels
        self.fc1 = nn.Linear(2 * hidden_channels, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, self.NUM_CLASSES)
        
        # Dropout for dense layers
        self.fc_dropout = nn.Dropout(dropout)
        
        # Activation for dense layers
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
        # Forward pass
        forward = self.forward_projection(x)  # (batch, hidden_channels, time_steps)
        for tcn_block in self.forward_tcn_blocks:
            forward = tcn_block(forward)
        
        # Get forward representation
        forward_repr = self.global_pool(forward).squeeze(-1)  # (batch, hidden_channels)
        
        # Reverse pass
        x_reversed = torch.flip(x, dims=[2])  # Reverse time dimension
        reverse = self.reverse_projection(x_reversed)
        for tcn_block in self.reverse_tcn_blocks:
            reverse = tcn_block(reverse)
        
        # Get reverse representation
        reverse_repr = self.global_pool(reverse).squeeze(-1)  # (batch, hidden_channels)
        
        # Concatenate representations
        combined = torch.cat([forward_repr, reverse_repr], dim=1)  # (batch, 2*hidden_channels)
        
        # Dense layers
        out = self.fc1(combined)  # (batch, 256)
        out = self.fc_activation(out)
        out = self.fc_dropout(out)
        
        out = self.fc2(out)  # (batch, 128)
        out = self.fc_activation(out)
        out = self.fc_dropout(out)
        
        out = self.fc3(out)  # (batch, NUM_CLASSES)
        
        return out
    
    def forward_with_attention(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with intermediate representations for analysis.
        
        Args:
            x: Input tensor of shape (batch, channels, time_steps)
            
        Returns:
            Tuple of (logits, bidirectional_features)
        """
        # Forward pass
        forward = self.forward_projection(x)
        for tcn_block in self.forward_tcn_blocks:
            forward = tcn_block(forward)
        forward_repr = self.global_pool(forward).squeeze(-1)
        
        # Reverse pass
        x_reversed = torch.flip(x, dims=[2])
        reverse = self.reverse_projection(x_reversed)
        for tcn_block in self.reverse_tcn_blocks:
            reverse = tcn_block(reverse)
        reverse_repr = self.global_pool(reverse).squeeze(-1)
        
        # Concatenate and classify
        combined = torch.cat([forward_repr, reverse_repr], dim=1)
        
        out = self.fc1(combined)
        out = self.fc_activation(out)
        out = self.fc_dropout(out)
        
        out = self.fc2(out)
        out = self.fc_activation(out)
        out = self.fc_dropout(out)
        
        logits = self.fc3(out)
        
        return logits, combined
    
    def get_class_names(self) -> List[str]:
        """Get fencing action class names."""
        return ["R", "IS", "WW", "JS", "SF", "SB"]
