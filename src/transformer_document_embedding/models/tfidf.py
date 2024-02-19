from __future__ import annotations
from typing import Iterator, Optional, TYPE_CHECKING

from gensim.corpora import Dictionary
from gensim.matutils import sparse2full
from gensim.models import TfidfModel
from transformer_document_embedding.datasets import col
from transformer_document_embedding.models.embedding_model import EmbeddingModel


import torch

if TYPE_CHECKING:
    from datasets import Dataset


class TFIDF(EmbeddingModel):
    def __init__(
        self,
        word_filter_no_below: int,
        word_filter_no_above: float,
        smartirs: str,
    ) -> None:
        self._word_filter_no_below = word_filter_no_below
        self._word_filter_no_above = word_filter_no_above

        self._smartirs = smartirs

    def predict_embeddings(self, dataset: Dataset) -> Iterator[torch.Tensor]:
        words_dataset = dataset.map(
            lambda doc: {"words": doc[col.TEXT].lower().split()},
            remove_columns=[col.ID, col.TEXT],
        )
        word_dict = self._get_word_dictionary(words_dataset)

        model = TfidfModel(
            smartirs=self._smartirs,
            dictionary=word_dict,
        )
        embed_dim = len(word_dict.keys())

        for article_words in words_dataset["words"]:
            sparse_vec = model[word_dict.doc2bow(article_words)]
            yield torch.from_numpy(sparse2full(sparse_vec, embed_dim))

    def _get_word_dictionary(
        self,
        dataset: Dataset,
        *,
        no_below: Optional[int] = None,
        no_above: Optional[float] = None,
    ) -> Dictionary:
        no_below = no_below if no_below is not None else self._word_filter_no_below
        no_above = no_above if no_above is not None else self._word_filter_no_above

        gensim_dict = Dictionary(dataset["words"])

        gensim_dict.filter_extremes(no_below=no_below, no_above=no_above)

        return gensim_dict

    def save_weights(self, path: str) -> None:
        pass

    def load_weights(self, path: str, *, strict: bool = False) -> None:
        pass
