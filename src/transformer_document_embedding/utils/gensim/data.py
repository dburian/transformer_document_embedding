from typing import Iterable
from datasets.arrow_dataset import Dataset
from datasets.iterable_dataset import IterableDataset
from gensim.models import doc2vec


class GensimCorpus:
    def __init__(self, dataset: Dataset | IterableDataset) -> None:
        self._dataset = dataset

    def __iter__(self) -> Iterable[doc2vec.TaggedDocument]:
        for doc in self._dataset:
            yield doc2vec.TaggedDocument(self.preprocess_text(doc["text"]), [doc["id"]])

    @classmethod
    def preprocess_text(cls, doc_text: str) -> list[str]:
        words = doc_text.split()

        return words
