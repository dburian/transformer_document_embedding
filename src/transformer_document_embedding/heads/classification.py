from __future__ import annotations
import torch
from transformer_document_embedding.datasets import col

from transformer_document_embedding.utils.net_helpers import get_activation
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformer_document_embedding.models.embedding_model import EmbeddingModel


class ClassificationHead(torch.nn.Module):
    def __init__(
        self,
        *,
        hidden_features: int,
        hidden_activation: str,
        hidden_dropout: float,
        out_features: int,
        label_smoothing: float,
        embedding_model: EmbeddingModel,
    ) -> None:
        super().__init__()

        in_features = self._get_embed_dim(embedding_model)

        layers = []
        if hidden_features > 0:
            layers.append(torch.nn.Linear(in_features, hidden_features))
            layers.append(get_activation(hidden_activation)())
            if hidden_dropout > 0:
                layers.append(torch.nn.Dropout(hidden_dropout))

        last_in_features = in_features if hidden_features == 0 else hidden_features
        layers.append(torch.nn.Linear(last_in_features, out_features))

        self.ff = torch.nn.Sequential(*layers)
        self.loss = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def _get_embed_dim(self, embedding_model: EmbeddingModel) -> int:
        return embedding_model.embedding_dim

    def forward(self, **kwargs) -> dict[str, torch.Tensor]:
        logits = self.ff(kwargs[col.EMBEDDING])

        outputs = {"logits": logits}
        if (labels := kwargs.get(col.LABEL, None)) is not None:
            outputs["loss"] = self.loss(logits, labels)

        return outputs


class PairClassificationHead(ClassificationHead):
    def _get_embed_dim(self, embedding_model: EmbeddingModel) -> int:
        return embedding_model.embedding_dim * 2
