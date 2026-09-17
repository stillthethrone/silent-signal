"""Classification heads for isolated sign recognition."""

from __future__ import annotations

from torch import Tensor, nn


class ClassificationHead(nn.Module):
    """Layer-normalized linear classification head."""

    def __init__(self, embedding_dim: int, num_classes: int) -> None:
        super().__init__()
        if embedding_dim < 1 or num_classes < 2:
            raise ValueError("embedding_dim must be positive and num_classes at least 2.")
        self.layers = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, num_classes),
        )

    def forward(self, embedding: Tensor) -> Tensor:
        return self.layers(embedding)
