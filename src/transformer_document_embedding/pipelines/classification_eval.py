from __future__ import annotations
from dataclasses import dataclass
import itertools
from typing import TYPE_CHECKING, Iterable, Iterator

from transformer_document_embedding.datasets import col
from transformer_document_embedding.pipelines.helpers import classification_metrics

from transformer_document_embedding.pipelines.pipeline import EvalPipeline
import torch

if TYPE_CHECKING:
    from datasets import Dataset
    from transformer_document_embedding.datasets.document_dataset import DocumentDataset
    from transformer_document_embedding.models.embedding_model import EmbeddingModel


def smart_unbatch(
    iterable: Iterable[torch.Tensor],
    single_dim: int,
) -> Iterator[torch.Tensor]:
    for batch in iterable:
        if len(batch.shape) > single_dim:
            yield from smart_unbatch(batch, single_dim)
        else:
            yield batch


@dataclass(kw_only=True)
class ClassificationEval(EvalPipeline):
    batch_size: int

    def get_embeddings_iter(
        self, split: Dataset, model: EmbeddingModel
    ) -> Iterator[torch.Tensor]:
        return model.predict_embeddings(split, batch_size=self.batch_size)

    @torch.inference_mode()
    def __call__(
        self,
        model: EmbeddingModel,
        head: torch.nn.Module,
        dataset: DocumentDataset,
    ) -> dict[str, float]:
        test_split = dataset.splits["test"]
        head.eval()

        embeddings_iter = self.get_embeddings_iter(test_split, model)
        peeked_batch = next(embeddings_iter)
        embeddings_iter = itertools.chain([peeked_batch], embeddings_iter)

        device = peeked_batch.device
        batch_size = peeked_batch.shape[0]

        num_classes = len(dataset.splits["test"].unique(col.LABEL))
        metrics = classification_metrics(num_classes, device=device)
        head.to(device)

        labels_batches = (
            doc[col.LABEL].to(device)
            for doc in test_split.with_format("torch").iter(batch_size)
        )

        for labels, embeddings in zip(labels_batches, embeddings_iter, strict=True):
            logits = head(**{col.EMBEDDING: embeddings})["logits"]
            pred_classes = torch.argmax(logits, dim=1)

            for metric in metrics.values():
                metric.update(pred_classes, labels)

        return {name: met.compute().item() for name, met in metrics.items()}


class PairClassificationEval(ClassificationEval):
    def get_embeddings_iter(
        self, split: Dataset, model: EmbeddingModel
    ) -> Iterator[torch.Tensor]:
        embeddings_0 = model.predict_embeddings(
            split.rename_columns({col.TEXT_0: col.TEXT, col.ID_0: col.ID}),
            batch_size=self.batch_size,
        )
        embeddings_1 = model.predict_embeddings(
            split.rename_columns({col.TEXT_1: col.TEXT, col.ID_1: col.ID}),
            batch_size=self.batch_size,
        )

        for embed_0, embed_1 in zip(embeddings_0, embeddings_1, strict=True):
            yield torch.concat((embed_0, embed_1), dim=1)
