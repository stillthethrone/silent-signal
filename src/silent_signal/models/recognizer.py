"""End-to-end pose graph recognizer."""

from __future__ import annotations

from torch import Tensor, nn

from silent_signal.models.heads.classification import ClassificationHead


class PoseGraphRecognizer(nn.Module):
    """Combine a reusable pose graph encoder with a 200-class head."""

    def __init__(
        self,
        encoder: nn.Module,
        *,
        embedding_dim: int,
        num_classes: int,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.classifier = ClassificationHead(embedding_dim, num_classes)

    def forward(
        self,
        features: Tensor,
        joint_mask: Tensor,
        frame_mask: Tensor,
        adjacency: Tensor,
    ) -> Tensor:
        embedding = self.encoder(features, joint_mask, frame_mask, adjacency)
        return self.classifier(embedding)
