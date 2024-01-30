from __future__ import annotations
from typing import TYPE_CHECKING, Iterable, Iterator, cast
import logging

from torch.utils.data import DataLoader, IterableDataset
from transformer_document_embedding.models.paragraph_vector.classifier import (
    ParagraphVectorClassifier,
)

from transformer_document_embedding.models.cls_head import ClsHead
from transformer_document_embedding.utils.gensim import (
    GensimCorpus,
    TextPreProcessor,
)
import torch
from typing import Any
import numpy as np
import datasets


if TYPE_CHECKING:
    from .paragraph_vector import ParagraphVector
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


class PairedGensimCorpus(GensimCorpus):
    """Gensim corpus for pairs of sentences as one item.

    E.g. for pairwise classification tasks, where PV needs each sentence separately."""

    def __init__(
        self,
        dataset: datasets.Dataset,
        text_pre_processor: TextPreProcessor,
        num_proc: int = 0,
    ) -> None:
        self._text_preprocessor = text_pre_processor
        self._dataset = datasets.Dataset.from_generator(
            self._unique_gensim_docs_iter,
            gen_kwargs={"pairs_dataset": dataset},
            num_proc=num_proc,
        )

    def _unique_gensim_docs_iter(
        self, pairs_dataset: datasets.Dataset
    ) -> Iterable[dict[str, Any]]:
        def _duplicates_iter() -> Iterable[dict[str, Any]]:
            for pair in pairs_dataset:
                pair = cast(dict[str, Any], pair)
                for i in range(2):
                    yield {
                        "words": self._text_preprocessor(pair[f"text_{i}"]),
                        "tag": pair[f"id_{i}"],
                    }

        seen_tags = set()
        for doc in _duplicates_iter():
            if doc["tag"] in seen_tags:
                continue
            seen_tags.add(doc["tag"])
            yield doc


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
                    doc[f"text_{i}"],
                    doc[f"id_{i}"],
                )
                for i in range(2)
            ]
            input = {
                "embeddings": torch.tensor(np.concatenate(embeddings, axis=0)),
            }

            if "label" in doc:
                input["labels"] = torch.tensor(doc["label"])

            yield input
