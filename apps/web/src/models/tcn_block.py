import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm


class TCNBlock(nn.Module):
    """
    FenceNet TCN block from Zhu et al. CVPRW 2022.

    Each block applies two weight-normalized dilated causal Conv1D layers,
    ReLU, spatial dropout, and a residual connection with an optional 1x1
    projection when the channel count changes.
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

        self.activation = activation if activation is not None else nn.ReLU()

        # Symmetric Conv1D padding followed by trimming keeps output length
        # unchanged while preventing future context from leaking into time t.
        padding = (kernel_size - 1) * dilation

        self.conv1 = weight_norm(nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding
        ))
        self.conv2 = weight_norm(nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding
        ))

        self.causal_pad = padding

        # Spatial dropout for temporal features drops entire channels.
        self.dropout = nn.Dropout1d(dropout)

        if in_channels != out_channels:
            self.res_projection = nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=1
            )
        else:
            self.res_projection = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with causal convolution and residual connection.

        Args:
            x: Input tensor of shape (batch, in_channels, time_steps)

        Returns:
            Output tensor of shape (batch, out_channels, time_steps)
        """
        residual = x

        out = self._causal_conv(self.conv1, x)
        out = self.activation(out)
        out = self.dropout(out)

        out = self._causal_conv(self.conv2, out)
        out = self.activation(out)
        out = self.dropout(out)

        if self.res_projection is not None:
            residual = self.res_projection(x)

        if residual.shape[2] > out.shape[2]:
            residual = residual[:, :, :out.shape[2]]

        out = self.activation(out + residual)
        return out

    def _causal_conv(
        self,
        conv: nn.Module,
        x: torch.Tensor
    ) -> torch.Tensor:
        """Apply padded Conv1D and trim the right side to preserve causality."""
        out = conv(x)
        if self.causal_pad > 0:
            out = out[:, :, :-self.causal_pad]
        return out
