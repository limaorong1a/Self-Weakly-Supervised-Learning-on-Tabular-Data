from __future__ import annotations

import torch
from torch import nn


class MLPClassifier(nn.Module):
    """Compact MLP for encoded tabular features with optional BatchNorm adaptation."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        depth: int = 3,
        dropout: float = 0.15,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        dim = input_dim
        for _ in range(depth):
            layers.extend(
                [
                    nn.Linear(dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            dim = hidden_dim
        self.encoder = nn.Sequential(*layers)
        self.classifier = nn.Linear(dim, num_classes)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        z = self.encoder(x)
        logits = self.classifier(z)
        if return_features:
            return logits, z
        return logits


def clone_model(model: nn.Module) -> nn.Module:
    import copy

    return copy.deepcopy(model)
