from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Iterator, Optional, cast, Any
from gensim.models import doc2vec
from nltk.stem import PorterStemmer
from torch.utils.data import IterableDataset
from typing import Callable


if TYPE_CHECKING:
    from transformer_document_embedding.models.paragraph_vector.paragraph_vector import (  # noqa: E501
        ParagraphVector,
    )
    from datasets.arrow_dataset import Dataset


class GensimCorpus:
    def __init__(
        self, dataset: Dataset, text_pre_processor: TextPreProcessor, num_proc: int = 0
    ) -> None:
        self._text_preprocessor = text_pre_processor
        self._dataset = dataset.map(self.doc_to_gensim_doc, num_proc=num_proc)

    def __len__(self) -> int:
        return len(self._dataset)

    def __iter__(self) -> Iterator[doc2vec.TaggedDocument]:
        for gensim_doc in self._dataset:
            # To get rid of warnings that doc can also be a list
            gensim_doc = cast(dict[str, Any], gensim_doc)

            yield doc2vec.TaggedDocument(gensim_doc["words"], [gensim_doc["id"]])

    def doc_to_gensim_doc(self, doc: dict[str, Any]) -> dict[str, Any]:
        return {
            "words": self._text_preprocessor(doc["text"]),
            "id": doc["id"],
        }


class PairedGensimCorpus(GensimCorpus):
    """Gensim corpus for pairs of sentences as one item.

    E.g. for pairwise classification tasks, where PV needs each sentence separately."""

    def __iter__(self) -> Iterator[doc2vec.TaggedDocument]:
        for doc in self._dataset:
            # To get rid of warnings that doc can also be a list
            doc = cast(dict[str, Any], doc)

            yield doc2vec.TaggedDocument(doc["words1"], [doc["id1"]])
            yield doc2vec.TaggedDocument(doc["words2"], [doc["id2"]])

    def doc_to_gensim_doc(self, doc: dict[str, Any]) -> dict[str, Any]:
        return {
            "words1": self._text_preprocessor(doc["text1"]),
            "words2": self._text_preprocessor(doc["text2"]),
            "id1": doc["id"] * 2,
            "id2": doc["id"] * 2 + 1,
        }


class IterableFeaturesDataset(IterableDataset):
    def __init__(
        self,
        docs: Dataset,
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

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        for doc in self._docs:
            # To get rid of warnings that doc can also be a list
            doc = cast(dict[str, Any], doc)

            embedding = (
                self._pv.get_vector(doc["id"])
                if self._lookup_vectors
                else self._pv.infer_vector(self._text_preprocessor(doc["text"]))
            )
            input = {
                "embeddings": torch.tensor(embedding),
            }

            if "label" in doc:
                input["labels"] = torch.tensor(doc["label"])

            yield input


TextPreProcessor = Callable[[str], list[str]]


def create_text_pre_processor(pre_process: Optional[str]) -> TextPreProcessor:
    lowercase = pre_process in ["lowercase", "stem"]
    stemer = PorterStemmer() if pre_process == "stem" else None

    def pre_processor(text: str) -> list[str]:
        words = (text.lower() if lowercase else text).split()
        if stemer is not None:
            words = [stemer.stem(w) for w in words]

        return words

    return pre_processor
