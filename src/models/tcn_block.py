"""
TCN Block - Temporal Convolutional Network block with causal and dilated convolutions.
"""

import torch
import torch.nn as nn
from typing import Optional


class TCNBlock(nn.Module):
    """
    Temporal Convolutional Network block with causal convolution and residual connection.
    
    Features:
    - Causal convolution (no future data leakage)
    - Dilated convolution for increased receptive field
    - Residual connection: output = Activation(input + f(input))
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.5,
        activation: nn.Module = None
    ):
        """
        Initialize TCN Block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Convolution kernel size
            dilation: Dilation factor for dilated convolution
            dropout: Dropout rate
            activation: Activation function (default: ReLU)
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        
        if activation is None:
            activation = nn.ReLU()
        self.activation = activation
        
        # Calculate padding for causal convolution
        # For causal convolution, we need padding on the left only
        padding = (kernel_size - 1) * dilation
        
        # Main convolution path
        self.conv1 = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding
        )
        self.conv2 = nn.Conv1d(
            out_channels, out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding
        )
        
        # Causal padding removal (remove future samples)
        self.causal_pad = padding
        
        # Batch normalization
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Residual projection if input/output channels differ
        self.res_projection = None
        if in_channels != out_channels:
            self.res_projection = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm1d(out_channels)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with causal convolution and residual connection.
        
        Args:
            x: Input tensor of shape (batch, in_channels, time_steps)
            
        Returns:
            Output tensor of shape (batch, out_channels, time_steps)
        """
        # Remember input for residual connection
        residual = x
        
        # First convolution with batch norm and activation
        out = self.conv1(x)
        # Remove future padding for causality
        out = out[:, :, :-self.causal_pad] if self.causal_pad > 0 else out
        out = self.bn1(out)
        out = self.activation(out)
        out = self.dropout(out)
        
        # Second convolution with batch norm
        out = self.conv2(out)
        # Remove future padding for causality
        out = out[:, :, :-self.causal_pad] if self.causal_pad > 0 else out
        out = self.bn2(out)
        out = self.dropout(out)
        
        # Residual connection
        if self.res_projection is not None:
            residual = self.res_projection(x)
            # Ensure same sequence length after causal padding removal
            if residual.shape[2] > out.shape[2]:
                residual = residual[:, :, :out.shape[2]]
        else:
            # Ensure same sequence length
            if residual.shape[2] > out.shape[2]:
                residual = residual[:, :, :out.shape[2]]
        
        # Add residual and apply activation
        out = self.activation(out + residual)
        
        return out
