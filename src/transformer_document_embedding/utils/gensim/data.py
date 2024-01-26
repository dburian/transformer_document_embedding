from __future__ import annotations

from typing import TYPE_CHECKING
from gensim.models import doc2vec
from nltk.stem import PorterStemmer

if TYPE_CHECKING:
    from datasets.arrow_dataset import Dataset
    from typing import Iterable, Any


class GensimCorpus:
    def __init__(
        self, dataset: Dataset, lowercase: bool, stem: bool, num_proc: int = 0
    ) -> None:
        self._lowercase = lowercase
        self._stemer = PorterStemmer() if stem else None
        self._dataset = dataset.map(self.doc_to_gensim_doc, num_proc=num_proc)

    def __iter__(self) -> Iterable[doc2vec.TaggedDocument]:
        for gensim_doc in self._dataset:
            yield doc2vec.TaggedDocument(gensim_doc["words"], [gensim_doc["id"]])

    def preprocess_text(self, doc_text: str) -> list[str]:
        words = (doc_text.lower() if self._lowercase else doc_text).split()
        if self._stemer is not None:
            words = [self._stemer.stem(w) for w in words]

        return words

    def doc_to_gensim_doc(self, doc: dict[str, Any]) -> dict[str, Any]:
        return {
            "words": self.preprocess_text(doc["text"]),
            "id": doc["id"],
        }


class PairedGensimCorpus(GensimCorpus):
    """Gensim corpus for pairs of sentences as one item.

    E.g. for pairwise classification tasks, where PV needs each sentence separately."""

    def __iter__(self) -> Iterable[doc2vec.TaggedDocument]:
        for doc in self._dataset:
            yield doc2vec.TaggedDocument(doc["words1"], [doc["id1"]])
            yield doc2vec.TaggedDocument(doc["words2"], [doc["id2"]])

    def doc_to_gensim_doc(self, doc: dict[str, Any]) -> dict[str, Any]:
        return {
            "words1": self.preprocess_text(doc["text1"]),
            "words2": self.preprocess_text(doc["text2"]),
            "id1": self.transform_id(doc["id"], 0),
            "id2": self.transform_id(doc["id"], 1),
        }

    @classmethod
    def transform_id(cls, pair_id: int, idx_in_pair: int) -> int:
        return pair_id * 2 + idx_in_pair
