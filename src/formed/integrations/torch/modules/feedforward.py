"""Feed-forward neural network modules for PyTorch models.

This module provides feed-forward network layers with support for
multiple layers, dropout, layer normalization, and residual connections.

Key Components:
    - FeedForward: Multi-layer feed-forward network

Features:
    - Configurable activation functions
    - Layer normalization
    - Dropout for regularization
    - Residual connections

Example:
    >>> from formed.integrations.torch.modules import FeedForward
    >>> import torch.nn as nn
    >>>
    >>> # Simple 3-layer feed-forward network
    >>> ffn = FeedForward(
    ...     input_dim=256,
    ...     hidden_dim=128,
    ...     output_dim=64,
    ...     num_layers=3,
    ...     dropout=0.1,
    ...     activation=nn.GELU()
    ... )

"""

import torch
import torch.nn as nn


class FeedForward(nn.Module):
    """Multi-layer feed-forward neural network.

    Args:
        input_dim: Input dimension.
        hidden_dim: Hidden layer dimension.
        output_dim: Output dimension.
        num_layers: Number of layers.
        dropout: Dropout rate (0 means no dropout).
        activation: Activation module (default: ReLU).
        layer_norm: Whether to apply layer normalization.

    Example:
        >>> ffn = FeedForward(
        ...     input_dim=256,
        ...     hidden_dim=128,
        ...     output_dim=64,
        ...     num_layers=2,
        ...     dropout=0.1
        ... )

    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        activation: nn.Module = nn.ReLU(),
        layer_norm: bool = False,
    ) -> None:
        super().__init__()

        layers = []
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            out_dim = output_dim if i == num_layers - 1 else hidden_dim

            layers.append(nn.Linear(in_dim, out_dim))

            if i < num_layers - 1:  # No activation/dropout/norm on last layer
                layers.append(activation)
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
                if layer_norm:
                    layers.append(nn.LayerNorm(out_dim))

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (..., input_dim).

        Returns:
            Output tensor of shape (..., output_dim).

        """
        return self.layers(x)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return super().__call__(x)
