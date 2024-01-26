from typing import Iterable
from datasets.arrow_dataset import Dataset
from gensim.models import doc2vec
from nltk.stem import PorterStemmer


class GensimCorpus:
    def __init__(self, dataset: Dataset, lowercase: bool, stem: bool) -> None:
        self._dataset = dataset
        self._lowercase = lowercase
        self._stemer = PorterStemmer() if stem else None

    def __iter__(self) -> Iterable[doc2vec.TaggedDocument]:
        for doc in self._dataset:
            yield doc2vec.TaggedDocument(self.preprocess_text(doc["text"]), [doc["id"]])

    def preprocess_text(self, doc_text: str) -> list[str]:
        words = (doc_text.lower() if self._lowercase else doc_text).split()
        if self._stemer is not None:
            words = [self._stemer.stem(w) for w in words]

        return words


class PairedGensimCorpus(GensimCorpus):
    """Gensim corpus for pairs of sentences as one item.

    E.g. for pairwise classification tasks, where PV needs each sentence separately."""

    def __iter__(self) -> Iterable[doc2vec.TaggedDocument]:
        for doc in self._dataset:
            for idx, key in enumerate(["text1", "text2"]):
                new_text = self.preprocess_text(doc[key])
                new_id = self.transform_id(doc["id"], idx)
                yield doc2vec.TaggedDocument(new_text, [new_id])

    @classmethod
    def transform_id(cls, pair_id: int, idx_in_pair: int) -> int:
        return pair_id * 2 + idx_in_pair
