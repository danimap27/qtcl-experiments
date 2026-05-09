"""Classical classification head variants."""

import torch
import torch.nn as nn
from typing import List


class ClassicalHead(nn.Module):
    """
    Configurable classical head.

    Variants supported via constructor:
        variant="linear"      → Linear(d, n_classes) only (no hidden)
        variant="mlp"         → Linear(d, hidden) -> ReLU -> Dropout -> Linear(hidden, n_classes)
        variant="mlp_deep"    → Two hidden layers (hidden -> hidden//2 -> n_classes)
    """

    def __init__(
        self,
        feature_dim: int,
        n_classes: int = 2,
        hidden_dim: int = 128,
        dropout: float = 0.3,
        variant: str = "mlp",
    ):
        super().__init__()
        self.variant = variant

        if variant == "linear":
            self.net = nn.Linear(feature_dim, n_classes)
        elif variant == "mlp":
            self.net = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, n_classes),
            )
        elif variant == "mlp_deep":
            h2 = max(hidden_dim // 2, n_classes)
            self.net = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, h2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(h2, n_classes),
            )
        else:
            raise ValueError(f"Unknown classical variant: {variant}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
