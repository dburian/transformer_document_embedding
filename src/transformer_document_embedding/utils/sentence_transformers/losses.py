from sentence_transformers import SentenceTransformer
import torch
from typing import Sequence


class BCELoss(torch.nn.Module):
    def __init__(
        self, sent_transformer: SentenceTransformer, label_smoothing: float = 0.0
    ) -> None:
        super().__init__()
        self._sent_transformer = sent_transformer
        self._loss_fn = torch.nn.BCELoss()
        self._alpha = label_smoothing

    def forward(
        self, inputs: Sequence[dict[str, torch.Tensor]], labels: torch.Tensor
    ) -> torch.Tensor:
        pred_labels = self._sent_transformer(inputs[0])["sentence_embedding"]
        pred_labels = pred_labels[:, 0]

        labels = labels.float()
        labels += self._alpha
        labels += -labels * 2 * self._alpha
        return self._loss_fn(pred_labels, labels)
