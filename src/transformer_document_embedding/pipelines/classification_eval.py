from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING

from transformer_document_embedding.datasets import col
from transformer_document_embedding.pipelines.classification_finetune import (
    get_default_features,
    get_pair_bin_cls_features,
)
from transformer_document_embedding.pipelines.helpers import classification_metrics

from transformer_document_embedding.pipelines.pipeline import EvalPipeline
import torch

from transformer_document_embedding.utils.training import batch_to_device

if TYPE_CHECKING:
    from datasets import Dataset
    from transformer_document_embedding.datasets.document_dataset import DocumentDataset
    from transformer_document_embedding.models.embedding_model import EmbeddingModel


@dataclass(kw_only=True)
class ClassificationEval(EvalPipeline):
    batch_size: int

    def get_features(self, split: Dataset, model: EmbeddingModel) -> Dataset:
        return get_default_features(split, model, batch_size=self.batch_size)

    @torch.inference_mode()
    def __call__(
        self,
        model: EmbeddingModel,
        head: torch.nn.Module,
        dataset: DocumentDataset,
    ) -> dict[str, float]:
        test_split = dataset.splits["test"]
        head.eval()

        features = self.get_features(test_split, model)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        num_classes = len(dataset.splits["test"].unique(col.LABEL))
        metrics = classification_metrics(num_classes, device=device)
        head.to(device)

        for docs in features.with_format("torch").iter(self.batch_size):
            embeds = {col.EMBEDDING: docs[col.EMBEDDING]}
            batch_to_device(embeds, device)

            logits = head(**embeds)["logits"]
            pred_classes = torch.argmax(logits, dim=1)

            for metric in metrics.values():
                metric.update(pred_classes, docs[col.LABEL].to(device))

        return {name: met.compute().item() for name, met in metrics.items()}


class PairClassificationEval(ClassificationEval):
    def get_features(self, split: Dataset, model: EmbeddingModel) -> Dataset:
        return get_pair_bin_cls_features(split, model, batch_size=self.batch_size)
