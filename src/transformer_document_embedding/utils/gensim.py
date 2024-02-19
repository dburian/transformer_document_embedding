from __future__ import annotations

from typing import TYPE_CHECKING, Iterator, Optional, cast, Any
from gensim.models import doc2vec
from nltk.stem import PorterStemmer
from typing import Callable

from transformer_document_embedding.datasets import col


if TYPE_CHECKING:
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

            yield doc2vec.TaggedDocument(gensim_doc["words"], [gensim_doc["tag"]])

    def doc_to_gensim_doc(self, doc: dict[str, Any]) -> dict[str, Any]:
        return {
            "words": self._text_preprocessor(doc[col.TEXT]),
            "tag": doc[col.ID],
        }


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
