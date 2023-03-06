from sentence_transformers import SentenceTransformer
import torch
from typing import Sequence


class BCELoss(torch.nn.Module):
    def __init__(self, sent_transformer: SentenceTransformer) -> None:
        super().__init__()
        self._sent_transformer = sent_transformer
        self._loss_fn = torch.nn.BCELoss()

    def forward(
        self, inputs: Sequence[dict[str, torch.Tensor]], labels: torch.Tensor
    ) -> torch.Tensor:
        pred_labels = self._sent_transformer(inputs[0])["sentence_embedding"]
        pred_labels = pred_labels[:, 0]
        return self._loss_fn(pred_labels, labels.float())
