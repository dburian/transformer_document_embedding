from __future__ import annotations
import torch
from transformer_document_embedding.datasets import col

from transformer_document_embedding.utils.net_helpers import get_activation


class ClassificationHead(torch.nn.Module):
    def __init__(
        self,
        *,
        in_features: int,
        hidden_features: int,
        hidden_activation: str,
        hidden_dropout: float,
        out_features: int,
        label_smoothing: float,
    ) -> None:
        super().__init__()

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

    def forward(self, **kwargs) -> dict[str, torch.Tensor]:
        logits = self.ff(kwargs[col.EMBEDDING])

        outputs = {"logits": logits}
        if (labels := kwargs.get(col.LABEL, None)) is not None:
            outputs["loss"] = self.loss(logits, labels)

        return outputs
