from __future__ import annotations
from typing import TYPE_CHECKING, Iterator, cast
import logging

from torch.utils.data import DataLoader, IterableDataset
from transformer_document_embedding.models.paragraph_vector.classifier import (
    ParagraphVectorClassifier,
)

from transformer_document_embedding.models.cls_head import ClsHead
from transformer_document_embedding.utils.gensim import (
    GensimCorpus,
    PairedGensimCorpus,
    TextPreProcessor,
)
import torch
from typing import Any
import numpy as np


if TYPE_CHECKING:
    from .paragraph_vector import ParagraphVector
    import datasets
    from typing import Optional

logger = logging.getLogger(__name__)


class ParagraphVectorPairClassifier(ParagraphVectorClassifier):
    """ParagraphVectorClassifier adjusted to classify paired texts.

    The only thing different from a classifier is that each input (containing a pair
    of texts) gets two embedding vectors. This difference requires changing only
    - how texts are passed to PV
    - how embeddings are obtained.

    Otherwise we got a model that classifies each input into two categories.

    """

    def __init__(
        self,
        dbow_kwargs: Optional[dict[str, Any]],
        dm_kwargs: Optional[dict[str, Any]],
        cls_head_kwargs: dict[str, Any],
        label_smoothing: float,
        pre_process: str,
        batch_size: int,
    ) -> None:
        super().__init__(
            dm_kwargs=dm_kwargs,
            dbow_kwargs=dbow_kwargs,
            pre_process=pre_process,
            cls_head_kwargs=cls_head_kwargs,
            label_smoothing=label_smoothing,
            batch_size=batch_size,
        )

        # We need classification head for pairs of vectors
        self._model.cls_head = ClsHead(
            **cls_head_kwargs,
            in_features=2 * self._pv.vector_size,
            out_features=2,
        )

    def _get_gensim_corpus(self, text_dataset: datasets.Dataset) -> GensimCorpus:
        return PairedGensimCorpus(
            text_dataset,
            text_pre_processor=self._text_pre_processor,
            num_proc=max(m.workers for m in self._pv.modules),
        )

    def _feature_dataloader(self, data: datasets.Dataset, training: bool) -> DataLoader:
        features_dataset = IterableFeaturesDataset(
            data.shuffle() if training else data,
            self._text_pre_processor,
            self._pv,
            lookup_vectors=training,
        )
        return DataLoader(
            features_dataset,
            batch_size=self._batch_size,
        )


class IterableFeaturesDataset(IterableDataset):
    """Though it could be just normal dataset with __getitem__ method. Probably
    this is more efficient in terms of disk reads."""

    def __init__(
        self,
        docs: datasets.Dataset,
        text_pre_processor: TextPreProcessor,
        pv: ParagraphVector,
        lookup_vectors: bool,
    ) -> None:
        super().__init__()

        self._docs = docs
        self._text_preprocessor = text_pre_processor
        self._pv = pv
        self._lookup_vectors = lookup_vectors

    def __len__(self) -> int:
        return len(self._docs)

    def _get_embedding(self, text: str, identifier: int) -> np.ndarray:
        return (
            self._pv.get_vector(identifier)
            if self._lookup_vectors
            else self._pv.infer_vector(self._text_preprocessor(text))
        )

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        for doc in self._docs:
            # To get rid of warnings that doc can also be a list
            doc = cast(dict[str, Any], doc)

            embeddings = [
                self._get_embedding(
                    doc[f"text{offset + 1}"],
                    doc["id"] * 2 + offset,
                )
                for offset in range(2)
            ]
            input = {
                "embeddings": torch.tensor(np.concatenate(embeddings, axis=0)),
            }

            if "label" in doc:
                input["labels"] = torch.tensor(doc["label"])

            yield input
