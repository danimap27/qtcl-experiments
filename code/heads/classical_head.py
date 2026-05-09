"""Classical MLP classification head."""

import torch
import torch.nn as nn


class ClassicalHead(nn.Module):
    """
    Two-layer MLP with ReLU activation and dropout.

    Architecture: Linear(feature_dim, hidden_dim) -> ReLU -> Dropout -> Linear(hidden_dim, n_classes)
    """

    def __init__(
        self,
        feature_dim: int,
        n_classes: int = 2,
        hidden_dim: int = 128,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
